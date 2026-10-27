"""
Auto-Learner and Continuous Feedback system for Sentiment SLM.

Allows the model to learn and evolve safely from user feedback and new slang:
1. Captures user corrections via UI / API into data/feedback.jsonl.
2. Evaluates text against FinancialRelevanceGate to prevent spam and data poisoning.
3. Automatically triggers fast local neural training (~15-25 seconds).
4. Strictly gates new weights against data/golden_set.json benchmark:
   if accuracy regresses below safety threshold, new weights are rolled back.
5. Hot-reloads updated model weights in-memory without application downtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fomo_meter.config import (
    FEEDBACK_FILE,
    GOLDEN_FILE,
    MAX_TEXT_LENGTH,
    RETRAIN_MIN_GOLDEN_ACC,
    SLM_ARCHITECTURE,
    SLM_DATASET,
    SLM_FEATURE_DIM,
    SLM_META,
    SLM_VERSION,
    SLM_WEIGHTS,
)
from fomo_meter.relevance_gate import FinancialRelevanceGate
from fomo_meter.slm import SentimentSLM
from fomo_meter.slm_features import featurize

LOGGER = logging.getLogger("fomo_meter.auto_learner")

LABELS = ["BUY", "SELL", "WAIT"]
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def forward(
    batch: np.ndarray,
    params: Dict[str, np.ndarray],
    dropout: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    z1 = batch @ params["W1"] + params["b1"]
    h1 = np.maximum(z1, 0.0)
    mask1 = None
    if dropout > 0 and rng is not None:
        mask1 = (rng.random(h1.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
        h1 = h1 * mask1
    z2 = h1 @ params["W2"] + params["b2"]
    h2 = np.maximum(z2, 0.0)
    mask2 = None
    if dropout > 0 and rng is not None:
        mask2 = (rng.random(h2.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
        h2 = h2 * mask2
    logits = h2 @ params["W3"] + params["b3"]
    cache = {"batch": batch, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "logits": logits}
    if mask1 is not None:
        cache["mask1"] = mask1
    if mask2 is not None:
        cache["mask2"] = mask2
    return logits, cache


def predict_proba(
    matrix: np.ndarray, params: Dict[str, np.ndarray], batch_size: int = 256
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, len(matrix), batch_size):
        logits, _ = forward(matrix[start:start + batch_size], params, 0.0, None)
        outputs.append(softmax(logits))
    return np.vstack(outputs) if outputs else np.zeros((0, len(LABELS)), dtype=np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: List[float] = []
    for index in range(len(LABELS)):
        tp = int(np.sum((y_pred == index) & (y_true == index)))
        fp = int(np.sum((y_pred == index) & (y_true != index)))
        fn = int(np.sum((y_pred != index) & (y_true == index)))
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_model(
    matrix: np.ndarray,
    y_true: np.ndarray,
    params: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    probabilities = predict_proba(matrix, params)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred),
        "predictions": y_pred,
        "probabilities": probabilities,
    }


def featurize_dataset(rows: List[Dict[str, str]], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        matrix[index] = featurize(row["text"], dim)
        labels[index] = LABEL_INDEX[row["label"]]
    return matrix, labels


def init_network_params(dim: int, hidden1: int, hidden2: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.normal(0.0, math.sqrt(2.0 / dim), (dim, hidden1))).astype(np.float32),
        "b1": np.zeros(hidden1, dtype=np.float32),
        "W2": (rng.normal(0.0, math.sqrt(2.0 / hidden1), (hidden1, hidden2))).astype(np.float32),
        "b2": np.zeros(hidden2, dtype=np.float32),
        "W3": (rng.normal(0.0, math.sqrt(2.0 / hidden2), (hidden2, len(LABELS)))).astype(np.float32),
        "b3": np.zeros(len(LABELS), dtype=np.float32),
    }


class AutoLearner:
    """Manages continuous feedback collection, verification, retraining, and hot-reload."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        slm: Optional[SentimentSLM] = None,
        relevance_gate: Optional[FinancialRelevanceGate] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else FEEDBACK_FILE.parent
        self.feedback_file = self.data_dir / "feedback.jsonl"
        self.golden_file = self.data_dir / "golden_set.json"
        self.dataset_file = self.data_dir / "slm_training.jsonl"
        self.weights_file = SLM_WEIGHTS
        self.meta_file = SLM_META
        self.slm = slm
        self.relevance_gate = relevance_gate or FinancialRelevanceGate(self.data_dir)
        self._lock = threading.RLock()
        self._is_training = False
        self._last_training_result: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- feedback
    def record_feedback(
        self,
        text: str,
        label: str,
        source: str = "user_feedback",
        auto_retrain_threshold: int = 0,
    ) -> Dict[str, Any]:
        """Validate, sanitize, and record user correction into feedback.jsonl."""
        label = (label or "").strip().upper()
        if label not in LABEL_INDEX:
            return {
                "ok": False,
                "error": f"Label tidak valid '{label}'. Pilih salah satu dari: BUY, SELL, WAIT.",
                "code": "INVALID_LABEL",
            }

        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Teks tidak boleh kosong.", "code": "EMPTY_TEXT"}

        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH].strip()

        # Integrity Gate: Anti-Poisoning & Relevance Check
        decision = self.relevance_gate.evaluate(text)
        if not decision.is_relevant:
            return {
                "ok": False,
                "error": f"Teks ditolak oleh gate relevansi finansial: {decision.summary}",
                "code": "NOT_RELEVANT",
            }

        sanitized_text = decision.sanitized_text or text
        norm_key = " ".join(sanitized_text.lower().split())

        with self._lock:
            existing = self._load_feedback_rows()
            for row in existing:
                if " ".join(row.get("text", "").lower().split()) == norm_key and row.get("label") == label:
                    return {
                        "ok": True,
                        "duplicate": True,
                        "message": "Contoh dengan label ini sudah tersimpan di memori auto-learn.",
                        "pending_count": len(existing),
                    }

            entry = {
                "timestamp": time.time(),
                "text": sanitized_text,
                "label": label,
                "source": source,
            }
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            with self.feedback_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total_pending = len(existing) + 1

        # Check if automatic retrain threshold met
        retrain_started = False
        if auto_retrain_threshold > 0 and total_pending >= auto_retrain_threshold and not self._is_training:
            self.retrain_in_background()
            retrain_started = True

        return {
            "ok": True,
            "duplicate": False,
            "message": f"Koreksi berhasil disimpan untuk {label}.",
            "pending_count": total_pending,
            "retrain_started": retrain_started,
        }

    def _load_feedback_rows(self) -> List[Dict[str, str]]:
        if not self.feedback_file.exists():
            return []
        rows: List[Dict[str, str]] = []
        with self.feedback_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = str(data.get("text") or "").strip()
                    label = str(data.get("label") or "").strip().upper()
                    if text and label in LABEL_INDEX:
                        rows.append({"text": text, "label": label, "source": data.get("source", "user_feedback")})
                except json.JSONDecodeError:
                    continue
        return rows

    def get_stats(self) -> Dict[str, Any]:
        """Return feedback count, label distribution, and training status."""
        rows = self._load_feedback_rows()
        counts = Counter(r["label"] for r in rows)
        return {
            "total_feedbacks": len(rows),
            "distribution": {label: counts.get(label, 0) for label in LABELS},
            "is_training": self._is_training,
            "last_result": self._last_training_result,
        }

    # ----------------------------------------------------------------- retrain
    def retrain(
        self,
        min_golden_accuracy: float = RETRAIN_MIN_GOLDEN_ACC,
        epochs: int = 120,
        batch_size: int = 32,
        learning_rate: float = 0.003,
        val_fraction: float = 0.15,
        patience: int = 20,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Retrain Sentiment SLM incorporating user feedback.
        Guarded by Golden Set benchmark: automatically rolls back if accuracy regresses.
        """
        if self._is_training:
            return {"ok": False, "error": "Training sedang berjalan di background.", "code": "BUSY"}

        with self._lock:
            self._is_training = True

        try:
            started = time.perf_counter()
            # 1. Load base training dataset
            base_rows: List[Dict[str, str]] = []
            if self.dataset_file.exists():
                with self.dataset_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if line:
                            try:
                                r = json.loads(line)
                                if r.get("text") and r.get("label") in LABEL_INDEX:
                                    base_rows.append({"text": r["text"], "label": r["label"], "source": r.get("source", "base")})
                            except json.JSONDecodeError:
                                pass

            # 2. Load user feedbacks
            feedback_rows = self._load_feedback_rows()

            # Merge and deduplicate (feedbacks override or complement base)
            combined_map: Dict[str, Dict[str, str]] = {}
            for row in base_rows:
                key = " ".join(row["text"].lower().split())
                combined_map[key] = row
            for row in feedback_rows:
                key = " ".join(row["text"].lower().split())
                combined_map[key] = row

            all_rows = list(combined_map.values())
            if len(all_rows) < 50:
                return {"ok": False, "error": f"Dataset terlalu kecil ({len(all_rows)} baris).", "code": "TOO_SMALL"}

            counts = Counter(r["label"] for r in all_rows)
            if any(label not in counts for label in LABELS):
                return {"ok": False, "error": f"Dataset harus memiliki semua label: {dict(counts)}", "code": "INCOMPLETE_LABELS"}

            # Stratified split
            rng = np.random.default_rng(seed)
            train_rows: List[Dict[str, str]] = []
            val_rows: List[Dict[str, str]] = []
            for label in LABELS:
                group = [r for r in all_rows if r["label"] == label]
                order = rng.permutation(len(group))
                group = [group[i] for i in order]
                val_count = max(1, int(round(len(group) * val_fraction)))
                val_rows.extend(group[:val_count])
                train_rows.extend(group[val_count:])
            rng.shuffle(train_rows)
            rng.shuffle(val_rows)

            x_train, y_train = featurize_dataset(train_rows, SLM_FEATURE_DIM)
            x_val, y_val = featurize_dataset(val_rows, SLM_FEATURE_DIM)

            # Class weights
            total = len(train_rows)
            class_weight = np.array([total / (len(LABELS) * counts[label]) for label in LABELS], dtype=np.float32)
            class_weight = class_weight / np.mean(class_weight)

            # Initialize weights
            params = init_network_params(SLM_FEATURE_DIM, 128, 64, seed)
            names = list(params.keys())
            moments = {name: np.zeros_like(params[name]) for name in names}
            velocities = {name: np.zeros_like(params[name]) for name in names}
            best_accuracy = -1.0
            best_epoch = 0
            best_params = {name: value.copy() for name, value in params.items()}

            step = 0
            beta1 = 0.9
            beta2 = 0.999
            weight_decay = 1e-4
            dropout = 0.15

            for epoch in range(1, epochs + 1):
                order = rng.permutation(len(train_rows))
                x_train = x_train[order]
                y_train = y_train[order]

                for start in range(0, len(train_rows), batch_size):
                    x_batch = x_train[start:start + batch_size]
                    y_batch = y_train[start:start + batch_size]
                    logits, cache = forward(x_batch, params, dropout, rng)
                    probabilities = softmax(logits)
                    b_size = len(y_batch)
                    weights = class_weight[y_batch]

                    dlogits = probabilities.copy()
                    dlogits[np.arange(b_size), y_batch] -= 1.0
                    dlogits *= (weights[:, None] / b_size)

                    grads: Dict[str, np.ndarray] = {}
                    grads["W3"] = cache["h2"].T @ dlogits
                    grads["b3"] = np.sum(dlogits, axis=0)
                    dh2 = dlogits @ params["W3"].T
                    dh2[cache["z2"] <= 0] = 0.0
                    if "mask2" in cache:
                        dh2 *= cache["mask2"]
                    grads["W2"] = cache["h1"].T @ dh2
                    grads["b2"] = np.sum(dh2, axis=0)
                    dh1 = dh2 @ params["W2"].T
                    dh1[cache["z1"] <= 0] = 0.0
                    if "mask1" in cache:
                        dh1 *= cache["mask1"]
                    grads["W1"] = x_batch.T @ dh1
                    grads["b1"] = np.sum(dh1, axis=0)

                    for name in names:
                        grad = grads[name]
                        if name.startswith("W"):
                            grad = grad + weight_decay * params[name]
                        moments[name] = beta1 * moments[name] + (1.0 - beta1) * grad
                        velocities[name] = beta2 * velocities[name] + (1.0 - beta2) * (grad * grad)
                        m_hat = moments[name] / (1.0 - beta1 ** (step + 1))
                        v_hat = velocities[name] / (1.0 - beta2 ** (step + 1))
                        params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
                    step += 1

                val_eval = evaluate_model(x_val, y_val, params)
                if val_eval["accuracy"] > best_accuracy:
                    best_accuracy = float(val_eval["accuracy"])
                    best_epoch = epoch
                    best_params = {name: value.copy() for name, value in params.items()}

                if epoch - best_epoch >= patience:
                    break

            final_val = evaluate_model(x_val, y_val, best_params)

            # 3. SAFETY GATE: Golden Benchmark Evaluation
            golden_accuracy = 1.0
            golden_total = 0
            golden_correct = 0
            if self.golden_file.exists():
                with self.golden_file.open("r", encoding="utf-8") as handle:
                    golden_rows = json.load(handle)
                eval_rows = [r for r in golden_rows if r.get("expected_stance") in LABEL_INDEX]
                if eval_rows:
                    golden_total = len(eval_rows)
                    matrix = np.vstack([featurize(r["text"], SLM_FEATURE_DIM) for r in eval_rows])
                    probs = predict_proba(matrix, best_params)
                    preds = np.argmax(probs, axis=1)
                    golden_correct = sum(
                        int(LABELS[pred] == r["expected_stance"])
                        for pred, r in zip(preds, eval_rows)
                    )
                    golden_accuracy = golden_correct / golden_total

            # Anti-Regression Gate: Rollback if benchmark drops
            if golden_accuracy < min_golden_accuracy:
                msg = (
                    f"Rollback otomatis: Akurasi benchmark golden set ({golden_accuracy:.1%}) "
                    f"turun di bawah ambang batas aman ({min_golden_accuracy:.1%}). "
                    f"Bobot model lama dipertahankan demi menjaga integritas."
                )
                LOGGER.warning(msg)
                result = {
                    "ok": False,
                    "error": msg,
                    "code": "GOLDEN_GATE_REJECTED",
                    "golden_accuracy": golden_accuracy,
                    "required_accuracy": min_golden_accuracy,
                }
                self._last_training_result = result
                return result

            # 4. Save and Hot-Reload (Atomic File Swap to prevent Windows file lock collisions)
            self.weights_file.parent.mkdir(parents=True, exist_ok=True)
            temp_weights = self.weights_file.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp_weights,
                W1=best_params["W1"], b1=best_params["b1"],
                W2=best_params["W2"], b2=best_params["b2"],
                W3=best_params["W3"], b3=best_params["b3"],
            )
            temp_weights.replace(self.weights_file)

            elapsed = round(time.perf_counter() - started, 2)
            meta = {
                "version": SLM_VERSION,
                "architecture": SLM_ARCHITECTURE,
                "feature_dim": SLM_FEATURE_DIM,
                "hidden1": 128,
                "hidden2": 64,
                "labels": LABELS,
                "accuracy": round(float(final_val["accuracy"]), 4),
                "macro_f1": round(float(final_val["macro_f1"]), 4),
                "golden_accuracy": round(float(golden_accuracy), 4),
                "golden_correct": golden_correct,
                "golden_total": golden_total,
                "samples": len(all_rows),
                "feedbacks_incorporated": len(feedback_rows),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "best_epoch": best_epoch,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "label_counts": dict(counts),
                "training_seconds": elapsed,
                "notes": f"Auto-learned with {len(feedback_rows)} user feedback examples. Safe rollback guard passed.",
            }
            temp_meta = self.meta_file.with_suffix(".tmp.json")
            with temp_meta.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            temp_meta.replace(self.meta_file)

            # HOT-RELOAD in memory!
            if self.slm:
                self.slm.load()

            result = {
                "ok": True,
                "message": "Model berhasil dilatih ulang dan dimuat seketika (hot-reloaded).",
                "accuracy": round(float(final_val["accuracy"]), 4),
                "golden_accuracy": round(float(golden_accuracy), 4),
                "samples": len(all_rows),
                "feedbacks": len(feedback_rows),
                "elapsed_s": elapsed,
            }
            self._last_training_result = result
            LOGGER.info("Auto-learn retraining completed: %s", result)
            return result

        except Exception as exc:
            LOGGER.exception("Auto-learn retraining failed: %s", exc)
            result = {"ok": False, "error": f"Retraining gagal: {exc}", "code": "EXCEPTION"}
            self._last_training_result = result
            return result
        finally:
            with self._lock:
                self._is_training = False

    def retrain_in_background(
        self, on_complete: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> threading.Thread:
        """Run retraining asynchronously in a background thread."""
        def _worker() -> None:
            res = self.retrain()
            if on_complete:
                try:
                    on_complete(res)
                except Exception as exc:
                    LOGGER.error("on_complete callback error: %s", exc)

        thread = threading.Thread(target=_worker, name="AutoLearnerThread", daemon=True)
        thread.start()
        return thread

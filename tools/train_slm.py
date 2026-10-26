"""
Train Sentiment SLM, the in-house finance sentiment model.

This is a from-scratch numpy implementation: hashed n-gram features, a two-layer
ReLU network, and a softmax head. It does not import PyTorch, TensorFlow, Ollama,
OpenAI, HuggingFace, or any pretrained model.

Run:
    python tools/build_slm_dataset.py
    python tools/train_slm.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fomo_meter.config import SLM_DATASET, SLM_FEATURE_DIM, SLM_META, SLM_WEIGHTS
from fomo_meter.slm_features import FEATURE_DIM, featurize

LABELS = ["BUY", "SELL", "WAIT"]
LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}
GOLDEN_FILE = ROOT / "data" / "golden_set.json"


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSONL rusak di baris {line_number}: {exc}") from exc
            text = str(row.get("text") or "").strip()
            label = str(row.get("label") or "").strip().upper()
            if not text or label not in LABEL_INDEX:
                continue
            rows.append({"text": text, "label": label, "source": str(row.get("source") or "unknown")})
    return rows


def stratified_split(
    rows: List[Dict[str, str]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = np.random.default_rng(seed)
    train: List[Dict[str, str]] = []
    val: List[Dict[str, str]] = []
    for label in LABELS:
        group = [row for row in rows if row["label"] == label]
        order = rng.permutation(len(group))
        group = [group[index] for index in order]
        val_count = max(1, int(round(len(group) * val_fraction)))
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def featurize_rows(rows: List[Dict[str, str]], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    labels = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        matrix[index] = featurize(row["text"], dim)
        labels[index] = LABEL_INDEX[row["label"]]
    return matrix, labels


def init_params(dim: int, hidden1: int, hidden2: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": (rng.normal(0.0, math.sqrt(2.0 / dim), (dim, hidden1))).astype(np.float32),
        "b1": np.zeros(hidden1, dtype=np.float32),
        "W2": (rng.normal(0.0, math.sqrt(2.0 / hidden1), (hidden1, hidden2))).astype(np.float32),
        "b2": np.zeros(hidden2, dtype=np.float32),
        "W3": (rng.normal(0.0, math.sqrt(2.0 / hidden2), (hidden2, len(LABELS)))).astype(np.float32),
        "b3": np.zeros(len(LABELS), dtype=np.float32),
    }


def forward(
    batch: np.ndarray,
    params: Dict[str, np.ndarray],
    dropout: float,
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
    cache = {
        "batch": batch, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "logits": logits,
    }
    if mask1 is not None:
        cache["mask1"] = mask1
    if mask2 is not None:
        cache["mask2"] = mask2
    return logits, cache


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


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
    return float(np.mean(scores))


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {}
    for actual_index, actual in enumerate(LABELS):
        matrix[actual] = {}
        for predicted_index, predicted in enumerate(LABELS):
            matrix[actual][predicted] = int(
                np.sum((y_true == actual_index) & (y_pred == predicted_index))
            )
    return matrix


def evaluate(
    matrix: np.ndarray,
    y_true: np.ndarray,
    params: Dict[str, np.ndarray],
) -> Dict[str, object]:
    probabilities = predict_proba(matrix, params)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
        "predictions": y_pred,
        "probabilities": probabilities,
    }


def train(args: argparse.Namespace) -> int:
    if not SLM_DATASET.exists():
        raise SystemExit(
            f"Dataset tidak ditemukan: {SLM_DATASET}. Jalankan tools/build_slm_dataset.py dulu."
        )

    rows = load_jsonl(SLM_DATASET)
    if len(rows) < 100:
        raise SystemExit(f"Dataset terlalu kecil ({len(rows)} baris).")
    counts = Counter(row["label"] for row in rows)
    if any(label not in counts for label in LABELS):
        raise SystemExit(f"Dataset harus punya ketiga label. Ditemukan: {dict(counts)}")

    train_rows, val_rows = stratified_split(rows, args.val_fraction, args.seed)
    x_train, y_train = featurize_rows(train_rows, FEATURE_DIM)
    x_val, y_val = featurize_rows(val_rows, FEATURE_DIM)

    # Balanced class weights; WAIT otherwise tends to be under-predicted.
    total = len(train_rows)
    class_weight = np.array(
        [total / (len(LABELS) * counts[label]) for label in LABELS], dtype=np.float32
    )
    class_weight = class_weight / np.mean(class_weight)

    params = init_params(FEATURE_DIM, args.hidden1, args.hidden2, args.seed)
    names = list(params.keys())
    moments = {name: np.zeros_like(params[name]) for name in names}
    velocities = {name: np.zeros_like(params[name]) for name in names}
    best_accuracy = -1.0
    best_epoch = 0
    best_params = {name: value.copy() for name, value in params.items()}
    history: List[Dict[str, float]] = []
    rng = np.random.default_rng(args.seed + 99)
    step = 0

    print(f"Sentiment SLM training: {len(train_rows)} train / {len(val_rows)} validation")
    print(f"Labels: {dict(counts)}")
    print(f"Architecture: {FEATURE_DIM} -> {args.hidden1} -> {args.hidden2} -> 3")
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_rows))
        x_train = x_train[order]
        y_train = y_train[order]
        epoch_loss = 0.0
        batches = 0

        for start in range(0, len(train_rows), args.batch_size):
            x_batch = x_train[start:start + args.batch_size]
            y_batch = y_train[start:start + args.batch_size]
            logits, cache = forward(x_batch, params, args.dropout, rng)
            probabilities = softmax(logits)
            batch_size = len(y_batch)
            weights = class_weight[y_batch]

            ce = -np.sum(np.log(np.maximum(probabilities[np.arange(batch_size), y_batch], 1e-9)) * weights) / batch_size
            l2 = 0.5 * args.weight_decay * (
                float(np.sum(params["W1"] ** 2))
                + float(np.sum(params["W2"] ** 2))
                + float(np.sum(params["W3"] ** 2))
            ) / max(1, len(train_rows))
            loss = float(ce + l2)
            epoch_loss += loss
            batches += 1

            dlogits = probabilities.copy()
            dlogits[np.arange(batch_size), y_batch] -= 1.0
            dlogits *= (weights[:, None] / batch_size)

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
                    grad = grad + args.weight_decay * params[name]
                moments[name] = args.beta1 * moments[name] + (1.0 - args.beta1) * grad
                velocities[name] = args.beta2 * velocities[name] + (1.0 - args.beta2) * (grad * grad)
                m_hat = moments[name] / (1.0 - args.beta1 ** (step + 1))
                v_hat = velocities[name] / (1.0 - args.beta2 ** (step + 1))
                params[name] -= args.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            step += 1

        train_eval = evaluate(x_train, y_train, params)
        val_eval = evaluate(x_val, y_val, params)
        history.append({
            "epoch": epoch,
            "loss": epoch_loss / max(1, batches),
            "train_accuracy": float(train_eval["accuracy"]),
            "val_accuracy": float(val_eval["accuracy"]),
            "val_macro_f1": float(val_eval["macro_f1"]),
        })

        improved = float(val_eval["accuracy"]) > best_accuracy
        if improved:
            best_accuracy = float(val_eval["accuracy"])
            best_epoch = epoch
            best_params = {name: value.copy() for name, value in params.items()}

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"epoch {epoch:3d} loss {epoch_loss / max(1, batches):.4f} "
                f"train {train_eval['accuracy']:.3f} val {val_eval['accuracy']:.3f} "
                f"f1 {val_eval['macro_f1']:.3f}"
            )
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
            break

    elapsed = time.perf_counter() - started
    final_train = evaluate(x_train, y_train, best_params)
    final_val = evaluate(x_val, y_val, best_params)

    # Golden benchmark is excluded from training. It is the external sanity check.
    golden_accuracy = None
    golden_total = 0
    golden_correct = 0
    if GOLDEN_FILE.exists():
        with GOLDEN_FILE.open("r", encoding="utf-8") as handle:
            golden_rows = json.load(handle)
        golden_eval_rows = [
            row for row in golden_rows
            if row.get("expected_stance") in LABEL_INDEX
        ]
        if golden_eval_rows:
            texts = [str(row["text"]) for row in golden_eval_rows]
            matrix = np.vstack([featurize(text, FEATURE_DIM) for text in texts])
            probabilities = predict_proba(matrix, best_params)
            predictions = np.argmax(probabilities, axis=1)
            golden_total = len(golden_eval_rows)
            golden_correct = 0
            print("\nGolden benchmark (external):")
            for row, prediction, probability in zip(golden_eval_rows, predictions, probabilities):
                predicted = LABELS[int(prediction)]
                expected = str(row["expected_stance"])
                ok = predicted == expected
                golden_correct += int(ok)
                print(
                    f"  {row.get('id')}: expected {expected:4s} predicted {predicted:4s} "
                    f"{'OK' if ok else 'MISS'} ({max(probability):.2f})"
                )
            golden_accuracy = golden_correct / golden_total
            print(f"Golden accuracy: {golden_correct}/{golden_total} = {golden_accuracy:.1%}")

    SLM_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SLM_WEIGHTS,
        W1=best_params["W1"], b1=best_params["b1"],
        W2=best_params["W2"], b2=best_params["b2"],
        W3=best_params["W3"], b3=best_params["b3"],
    )

    dataset_bytes = SLM_DATASET.read_bytes()
    meta = {
        "version": "Sentiment SLM",
        "architecture": f"Hashed n-gram features ({FEATURE_DIM}) -> {args.hidden1} -> {args.hidden2} -> 3 (softmax)",
        "feature_dim": FEATURE_DIM,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "labels": LABELS,
        "accuracy": round(float(final_val["accuracy"]), 4),
        "macro_f1": round(float(final_val["macro_f1"]), 4),
        "golden_accuracy": round(float(golden_accuracy), 4) if golden_accuracy is not None else None,
        "golden_correct": golden_correct,
        "golden_total": golden_total,
        "samples": len(rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "label_counts": dict(counts),
        "class_weight": {label: round(float(weight), 4) for label, weight in zip(LABELS, class_weight)},
        "final_train_accuracy": round(float(final_train["accuracy"]), 4),
        "final_val_confusion": final_val["confusion"],
        "training_seconds": round(elapsed, 2),
        "notes": (
            "Trained locally from data/slm_training.jsonl. "
            "No pretrained weights or hosted model APIs are used."
        ),
    }
    with SLM_META.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (ROOT / "models" / "sentiment_slm_training.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nSaved {SLM_WEIGHTS} ({SLM_WEIGHTS.stat().st_size / 1024:.0f} KB)")
    print(f"Saved {SLM_META}")
    print(f"Validation accuracy: {final_val['accuracy']:.1%} | macro F1: {final_val['macro_f1']:.3f}")
    print(f"Training time: {elapsed:.1f}s | best epoch: {best_epoch}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sentiment SLM from scratch.")
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))

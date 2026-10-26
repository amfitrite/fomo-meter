"""
SLM-1: the in-house neural sentiment model.

This module loads a small locally-trained network from models/slm1.npz. It is
NOT a wrapper around Ollama, OpenAI, HuggingFace, or any hosted LLM. The model
is trained by tools/train_slm.py on the project's own curated dataset.

Architecture (default):
    hashed n-gram features (2048) -> 128 ReLU -> 64 ReLU -> 3 softmax

The app can always fall back to the deterministic lexicon classifier when the
model is absent or uncertain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from sentiment_llm.config import (
    SLM_ARCHITECTURE,
    SLM_FEATURE_DIM,
    SLM_META,
    SLM_VERSION,
    SLM_WEIGHTS,
)
from sentiment_llm.schema import ModelStatus, Stance
from sentiment_llm.slm_features import featurize


class SLMUnavailable(RuntimeError):
    pass


@dataclass
class SLMPrediction:
    stance: Stance
    intensity: int
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model: str = SLM_VERSION
    latency_ms: int = 0


class SLM1:
    LABELS = ("BUY", "SELL", "WAIT")

    def __init__(
        self,
        weights_path: str | Path = SLM_WEIGHTS,
        meta_path: str | Path = SLM_META,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.meta_path = Path(meta_path)
        self.available = False
        self.error: Optional[str] = None
        self.version = SLM_VERSION
        self.architecture = SLM_ARCHITECTURE
        self.accuracy = 0.0
        self.samples = 0
        self.trained_at: Optional[str] = None
        self.feature_dim = SLM_FEATURE_DIM
        self.labels: List[str] = list(self.LABELS)
        self._params: Dict[str, np.ndarray] = {}
        self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        try:
            if not self.weights_path.exists():
                self.error = (
                    f"Bobot model tidak ditemukan: {self.weights_path.name}. "
                    "Jalankan python tools/train_slm.py untuk membangun SLM-1."
                )
                self.available = False
                return False
            if not self.meta_path.exists():
                self.error = f"Metadata model tidak ditemukan: {self.meta_path.name}."
                self.available = False
                return False

            with self.meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.version = str(meta.get("version") or SLM_VERSION)
            self.architecture = str(meta.get("architecture") or SLM_ARCHITECTURE)
            self.accuracy = float(meta.get("accuracy") or 0.0)
            self.samples = int(meta.get("samples") or 0)
            self.trained_at = meta.get("trained_at")
            self.feature_dim = int(meta.get("feature_dim") or SLM_FEATURE_DIM)
            labels = list(meta.get("labels") or self.LABELS)
            if set(labels) != set(self.LABELS):
                raise ValueError("Label model tidak sesuai (harus BUY, SELL, WAIT).")

            with np.load(self.weights_path, allow_pickle=False) as archive:
                params = {key: archive[key].astype(np.float32) for key in archive.files}

            expected = {"W1", "b1", "W2", "b2", "W3", "b3"}
            if not expected.issubset(params):
                raise ValueError("Bobot model tidak lengkap.")
            w1 = params["W1"]
            if w1.shape[0] != self.feature_dim:
                raise ValueError(
                    f"Dimensi fitur tidak cocok: model {w1.shape[0]}, app {self.feature_dim}."
                )
            self._params = params
            self.labels = labels
            self.available = True
            self.error = None
            return True
        except Exception as exc:
            self.available = False
            self.error = f"Gagal memuat SLM-1: {type(exc).__name__}: {exc}"
            self._params = {}
            return False

    # --------------------------------------------------------------- status
    def status(self) -> ModelStatus:
        return ModelStatus(
            available=self.available,
            version=self.version,
            architecture=self.architecture,
            accuracy=round(float(self.accuracy), 4),
            samples=self.samples,
            trained_at=self.trained_at,
            labels=list(self.labels),
            feature_dim=self.feature_dim,
            error=self.error,
        )

    # -------------------------------------------------------------- predict
    def predict(self, text: str) -> SLMPrediction:
        if not self.available:
            raise SLMUnavailable(self.error or "SLM-1 tidak tersedia.")
        if not text or not text.strip():
            return SLMPrediction(
                stance=Stance.WAIT,
                intensity=50,
                confidence=0.0,
                probabilities={"BUY": 0.0, "SELL": 0.0, "WAIT": 1.0},
                model=self.version,
            )

        started = time.perf_counter()
        vector = featurize(text, self.feature_dim)
        w1, b1 = self._params["W1"], self._params["b1"]
        w2, b2 = self._params["W2"], self._params["b2"]
        w3, b3 = self._params["W3"], self._params["b3"]

        hidden1 = np.maximum(0.0, vector @ w1 + b1)
        hidden2 = np.maximum(0.0, hidden1 @ w2 + b2)
        logits = hidden2 @ w3 + b3

        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        probs = exp / float(np.sum(exp))

        index = int(np.argmax(probs))
        label = self.labels[index]
        stance = Stance(label) if label in Stance.__members__ else Stance.WAIT
        confidence = float(probs[index])

        if stance == Stance.WAIT:
            intensity = 50
        else:
            intensity = int(round(50 + abs(float(probs[self.labels.index("BUY")]) - float(probs[self.labels.index("SELL")])) * 50))
            intensity = max(51, min(99, intensity))

        probability_map = {
            label_name: round(float(probability), 4)
            for label_name, probability in zip(self.labels, probs)
        }
        latency = int((time.perf_counter() - started) * 1000)
        return SLMPrediction(
            stance=stance,
            intensity=intensity,
            confidence=round(confidence, 4),
            probabilities=probability_map,
            model=self.version,
            latency_ms=latency,
        )

    def predict_many(self, texts: List[str]) -> List[SLMPrediction]:
        return [self.predict(text) for text in texts]

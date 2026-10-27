"""
Feature extraction for Sentiment SLM, the in-house neural sentiment model.

Features are stable hashed n-grams:
- word unigrams and bigrams
- character 3/4/5-grams for slang and morphology
- emoji tokens
- negation markers and the token following a negation
- length/question features

The exact same featurizer is used by tools/train_slm.py and at inference, so
there is no vocabulary drift between training and production.
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, Tuple

import numpy as np

from fomo_meter.classifier import normalize_text

FEATURE_DIM = 2048

NEGATIONS = {
    "not", "no", "never", "none", "cannot", "can't", "cant", "don't", "dont",
    "doesn't", "doesnt", "didn't", "didnt", "won't", "wont", "isn't", "isnt",
    "aren't", "arent", "wasn't", "wasnt", "without", "jangan", "bukan", "gak",
    "ga", "nggak", "ngga", "tidak", "tak", "tanpa", "belum",
}

_WORD_RE = re.compile(r"[a-z']+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CASHTAG_RE = re.compile(r"\$([a-z0-9.]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:k|m|b|%|x)?\b", re.IGNORECASE)


# Google-style Sarcasm & Contrastive Incongruity Markers ---------------------
HYPER_BULLISH_MARKERS = (
    "all in", "to the moon", "moon", "generational wealth", "easy money",
    "cant lose", "cannot lose", "financial advice", "guaranteed", "buy the top",
    "serok pucuk", "allin", "never selling", "diamond hands",
)

SARCASM_MOCKERY_MARKERS = (
    "clown", "clowns", "🤡", "rip", "💀", "hfsp", "have fun staying poor",
    "copium", "exit liquidity", "bagholder", "bagholders", "top buyer",
    "top buyers", "lol", "lmao", "rofl", "sure buddy", "good luck with that",
)


def tokenize(text: str) -> Tuple[List[str], List[str]]:
    """Return (word tokens, emoji tokens)."""
    normalized = normalize_text(text)
    lower = normalized.lower()
    lower = _URL_RE.sub(" url ", lower)
    lower = _CASHTAG_RE.sub(lambda match: " cashtag_" + match.group(1).replace(".", "_") + " ", lower)
    lower = _NUMBER_RE.sub(" <num> ", lower)
    words = _WORD_RE.findall(lower)
    emojis = [
        char for char in normalized
        if ord(char) > 0x2100 and not char.isspace() and char not in "\u200d\ufe0f"
    ]
    return words, emojis


def _add(features: Dict[str, float], name: str, value: float) -> None:
    if value == 0:
        return
    features[name] = features.get(name, 0.0) + value


def feature_weights(text: str) -> Dict[str, float]:
    words, emojis = tokenize(text)
    features: Dict[str, float] = {}

    for word in words:
        _add(features, f"u:{word}", 1.0)

    for left, right in zip(words, words[1:]):
        _add(features, f"b:{left}_{right}", 1.15)

    for word in words:
        if len(word) < 3:
            continue
        padded = f"^{word}$"
        for size in (3, 4, 5):
            if len(padded) < size:
                continue
            for start in range(0, len(padded) - size + 1):
                _add(features, f"c{size}:{padded[start:start + size]}", 0.55)

    for emoji in emojis:
        _add(features, f"e:{emoji}", 1.4)

    for index, word in enumerate(words):
        if word in NEGATIONS:
            _add(features, "neg", 1.2)
            if index + 1 < len(words):
                _add(features, f"neg_next:{words[index + 1]}", 1.5)
            if index + 2 < len(words):
                _add(features, f"neg_next2:{words[index + 2]}", 0.8)

    length = len(words)
    bucket = "short" if length <= 4 else ("medium" if length <= 12 else "long")
    _add(features, f"len:{bucket}", 0.5)
    _add(features, "question" if "?" in text else "statement", 0.25)

    # Google-style Contextual Incongruity Sarcasm Detection ----------------
    lower_raw = text.lower()
    has_hyper_bull = any(marker in lower_raw for marker in HYPER_BULLISH_MARKERS)
    has_mockery = any(marker in lower_raw for marker in SARCASM_MOCKERY_MARKERS)
    if has_hyper_bull and has_mockery:
        _add(features, "sarcasm:mockery_bullish", 3.2)
        _add(features, "sarcasm:incongruity", 2.6)
        _add(features, "neg:bullish_intent", 2.2)

    return features


def featurize(text: str, dim: int = FEATURE_DIM) -> np.ndarray:
    """Hash features into a signed, L2-normalised float32 vector."""
    vector = np.zeros(dim, dtype=np.float32)
    for name, value in feature_weights(text).items():
        digest = zlib.crc32(name.encode("utf-8"))
        index = digest % dim
        sign = 1.0 if ((digest >> 16) & 1) else -1.0
        vector[index] += sign * value
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector

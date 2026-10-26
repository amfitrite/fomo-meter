"""
Deterministic Layer-1 classifier for FOMO Meter.

The classifier is deliberately lightweight, offline, and explainable:
- phrase-first matching (so "bull trap" is not counted as "bull")
- negation window and intensifiers
- finance-specific emoji polarity
- explicit ambiguity when buy/sell evidence is close

It is not an LLM. The optional neural model (Sentiment SLM) lives in slm.py and is
clearly labelled in the UI.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fomo_meter.schema import Stance


@dataclass
class Classification:
    stance: Stance
    intensity: int
    confidence: float
    matched_terms: List[str] = field(default_factory=list)
    evidence_quote: Optional[str] = None
    is_ambiguous: bool = False
    buy_score: int = 0
    sell_score: int = 0
    wait_score: int = 0
    hit_count: int = 0

    @property
    def scores(self) -> Dict[str, int]:
        return {"BUY": self.buy_score, "SELL": self.sell_score, "WAIT": self.wait_score}


DEFAULT_TERMS: Dict[str, Dict[str, Any]] = {
    "buy the dip": {"stance": "BUY", "intensity": 85},
    "bull trap": {"stance": "SELL", "intensity": 75},
    "bullish": {"stance": "BUY", "intensity": 80},
    "bearish": {"stance": "SELL", "intensity": 80},
    "breakout": {"stance": "BUY", "intensity": 75},
    "breakdown": {"stance": "SELL", "intensity": 75},
    "sideways": {"stance": "WAIT", "intensity": 50},
    "wait and see": {"stance": "WAIT", "intensity": 50},
    "serok": {"stance": "BUY", "intensity": 85},
    "boncos": {"stance": "SELL", "intensity": 85},
}

DEFAULT_EMOJIS: Dict[str, Dict[str, Any]] = {
    "🚀": {"stance": "BUY", "intensity": 90},
    "📈": {"stance": "BUY", "intensity": 80},
    "🩸": {"stance": "SELL", "intensity": 90},
    "📉": {"stance": "SELL", "intensity": 85},
    "🦀": {"stance": "WAIT", "intensity": 55},
}

DEFAULT_NEGATIONS = {
    "not", "no", "never", "none", "cannot", "can't", "cant", "don't", "dont",
    "doesn't", "doesnt", "didn't", "didnt", "won't", "wont", "isn't", "isnt",
    "aren't", "arent", "wasn't", "wasnt", "without", "jangan", "bukan", "gak",
    "ga", "nggak", "ngga", "tidak", "tak", "tanpa", "belum",
}

DEFAULT_INTENSIFIERS = {
    "very": 1.25, "really": 1.20, "super": 1.30, "massive": 1.30,
    "huge": 1.25, "strong": 1.20, "extremely": 1.35, "parah": 1.25,
    "banget": 1.30, "besar": 1.20, "kuat": 1.20,
}

WORD_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize unicode punctuation and whitespace without destroying emojis."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class FastCrowdClassifier:
    """Phrase-first deterministic finance sentiment classifier."""

    def __init__(self, data_dir: Optional[str | Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.kb_path = self.data_dir / "slang_knowledge_base.json"
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.emojis: Dict[str, Dict[str, Any]] = {}
        self.negations = set(DEFAULT_NEGATIONS)
        self.intensifiers = dict(DEFAULT_INTENSIFIERS)
        self._patterns: List[Tuple[str, re.Pattern[str]]] = []
        self._load_kb()

    # ------------------------------------------------------------------ load
    def _load_kb(self) -> None:
        data: Dict[str, Any] = {}
        try:
            if self.kb_path.exists():
                with self.kb_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
        except (OSError, ValueError):
            data = {}

        self.terms = data.get("terms") or DEFAULT_TERMS
        self.emojis = data.get("emojis") or DEFAULT_EMOJIS
        self.negations = set(data.get("negations") or DEFAULT_NEGATIONS)
        self.intensifiers = dict(data.get("intensifiers") or DEFAULT_INTENSIFIERS)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        # Longest phrase first; the scanner also enforces non-overlapping spans.
        ordered = sorted(self.terms, key=lambda term: (-len(term), term))
        self._patterns = []
        for term in ordered:
            escaped = re.escape(term.strip().lower())
            if not escaped:
                continue
            pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
            self._patterns.append((term.lower(), pattern))

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _preceding_tokens(text: str, start: int, limit: int = 45) -> List[str]:
        before = text[max(0, start - limit):start]
        return WORD_RE.findall(before)

    def _is_negated(self, text: str, start: int) -> bool:
        tokens = self._preceding_tokens(text, start)
        return any(token in self.negations for token in tokens[-3:])

    def _intensifier_multiplier(self, text: str, start: int) -> float:
        tokens = self._preceding_tokens(text, start, limit=30)
        for token in reversed(tokens[-2:]):
            if token in self.intensifiers:
                return float(self.intensifiers[token])
        return 1.0

    def _collect_term_hits(self, lower_text: str) -> List[Tuple[int, int, str, Dict[str, Any]]]:
        """Return non-overlapping hits, longest phrases consuming shorter ones."""
        occupied: List[Tuple[int, int]] = []
        hits: List[Tuple[int, int, str, Dict[str, Any]]] = []

        for term, pattern in self._patterns:
            for match in pattern.finditer(lower_text):
                start, end = match.span()
                if any(not (end <= used_start or start >= used_end) for used_start, used_end in occupied):
                    continue
                occupied.append((start, end))
                hits.append((start, end, term, self.terms.get(term, {})))

        hits.sort(key=lambda item: item[0])
        return hits

    # ---------------------------------------------------------------- public
    def classify(self, text: str) -> Classification:
        original = normalize_text(text)
        if not original:
            return Classification(
                stance=Stance.WAIT, intensity=50, confidence=0.0,
                evidence_quote=None, is_ambiguous=True,
            )

        lower_text = original.lower()
        buy_score = 0.0
        sell_score = 0.0
        wait_score = 0.0
        matched: List[str] = []
        hit_count = 0

        # Terms and phrases ---------------------------------------------------
        term_counts: Dict[str, int] = {}
        for start, _end, term, meta in self._collect_term_hits(lower_text):
            term_counts[term] = term_counts.get(term, 0) + 1
            # Repeating a term more than three times should not dominate.
            if term_counts[term] > 3:
                continue

            stance = str(meta.get("stance", "WAIT")).upper()
            base_intensity = int(meta.get("intensity", 50))
            multiplier = self._intensifier_multiplier(lower_text, start)
            effective = base_intensity * multiplier

            if self._is_negated(lower_text, start):
                if stance == "BUY":
                    stance = "SELL"
                elif stance == "SELL":
                    stance = "BUY"

            if stance == "BUY":
                buy_score += effective
            elif stance == "SELL":
                sell_score += effective
            else:
                wait_score += effective

            if term not in matched:
                matched.append(term)
            hit_count += 1

        # Emojis --------------------------------------------------------------
        emoji_counts: Dict[str, int] = {}
        for emoji, meta in self.emojis.items():
            count = original.count(emoji)
            if count <= 0:
                continue
            emoji_counts[emoji] = count
            stance = str(meta.get("stance", "WAIT")).upper()
            effective = int(meta.get("intensity", 50)) * min(count, 3)
            if stance == "BUY":
                buy_score += effective
            elif stance == "SELL":
                sell_score += effective
            else:
                wait_score += effective
            if emoji not in matched:
                matched.append(emoji)
            hit_count += 1

        total = buy_score + sell_score + wait_score
        if total <= 0:
            return Classification(
                stance=Stance.WAIT,
                intensity=50,
                confidence=0.0,
                matched_terms=[],
                evidence_quote=original[:280],
                is_ambiguous=True,
                hit_count=0,
            )

        scores = {"BUY": buy_score, "SELL": sell_score, "WAIT": wait_score}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_name, top_value = ranked[0]
        second_value = ranked[1][1]

        # A bull/bear tie is genuinely ambiguous, not a forced direction.
        buy_sell_tie = abs(buy_score - sell_score) < 1.0 and buy_score > 0 and sell_score > 0
        if buy_sell_tie:
            stance = Stance.WAIT
            is_ambiguous = True
        else:
            stance = Stance(top_name)
            is_ambiguous = (
                buy_score > 0 and sell_score > 0
                and min(buy_score, sell_score) / max(buy_score, sell_score) >= 0.60
            )

        intensity = 50
        if stance in (Stance.BUY, Stance.SELL):
            margin = (top_value - second_value) / total
            intensity = int(round(50 + margin * 50))
            cap = 95 if hit_count == 1 else 99
            intensity = max(51, min(cap, intensity))

        share = top_value / total
        confidence = 0.45 + (0.35 * share) + (0.20 * min(1.0, hit_count / 3.0))
        if is_ambiguous:
            confidence -= 0.12
        confidence = round(max(0.0, min(0.98, confidence)), 2)

        return Classification(
            stance=stance,
            intensity=intensity,
            confidence=confidence,
            matched_terms=matched[:12],
            evidence_quote=original[:280],
            is_ambiguous=is_ambiguous,
            buy_score=int(round(buy_score)),
            sell_score=int(round(sell_score)),
            wait_score=int(round(wait_score)),
            hit_count=hit_count,
        )

    def explain(self, text: str) -> Dict[str, Any]:
        """Convenience payload for API/UI."""
        result = self.classify(text)
        return {
            "stance": result.stance.value,
            "intensity": result.intensity,
            "confidence": result.confidence,
            "matched_terms": result.matched_terms,
            "is_ambiguous": result.is_ambiguous,
            "scores": result.scores,
        }

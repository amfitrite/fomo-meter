"""
Anti-Hallucination verification gates.

The doctrine is simple: a model may analyse validated data, but it may never
become the source of that data. Missing, stale, mismatched, demo, or
unverifiable evidence is surfaced honestly instead of being filled with
synthetic consensus.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

from sentiment_llm.schema import FeedStatus, Stance


class AntiHallucinationVerifier:
    STALE_THRESHOLD_SECONDS = 60.0

    @classmethod
    def verify_feed(
        cls,
        raw_dict: Optional[Dict],
        expected_ticker: str,
        current_time: Optional[float] = None,
    ) -> Tuple[FeedStatus, Optional[str]]:
        """Legacy feed verification contract retained from the blueprint."""
        now = current_time if current_time is not None else time.time()

        # Gate 1: absence of data.
        if raw_dict is None or not raw_dict:
            return FeedStatus.NO_DATA, "Data file or payload absent. Fallback to demo prohibited."

        # Gate 4: structural integrity.
        for field in ("ticker", "timestamp", "sources"):
            if field not in raw_dict:
                return FeedStatus.INVALID, f"Corrupt payload: missing required field '{field}'"

        # Gate 3: symbol alignment.
        payload_ticker = str(raw_dict.get("ticker", "")).upper()
        if payload_ticker != expected_ticker.upper():
            return FeedStatus.SYMBOL_MISMATCH, (
                f"Ticker mismatch: expected {expected_ticker}, got {payload_ticker}"
            )

        # Gate 5: demo isolation.
        if raw_dict.get("is_demo", False):
            return FeedStatus.DEMO, "Payload contains synthetic or demonstration data."

        # Gates 2 and 9: freshness and heartbeat.
        try:
            timestamp = float(raw_dict.get("timestamp", 0.0))
        except (TypeError, ValueError):
            return FeedStatus.INVALID, "Invalid timestamp format."

        if (now - timestamp) > cls.STALE_THRESHOLD_SECONDS:
            return FeedStatus.STALE, (
                f"Feed age ({now - timestamp:.1f}s) exceeds stale threshold "
                f"({cls.STALE_THRESHOLD_SECONDS:.0f}s)."
            )

        return FeedStatus.LIVE, None

    # ------------------------------------------------------------ evidence
    @staticmethod
    def _normalize_quote(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip().lower()

    @classmethod
    def verify_evidence_quote(cls, quote: Optional[str], source_posts: Iterable[str]) -> bool:
        """
        Gate 7: the evidence quote must be an exact substring of a source post.

        Whitespace is normalised because HTML/RSS feeds wrap lines, but words
        may not be paraphrased, reordered, or invented.
        """
        if not quote or not quote.strip():
            return False
        clean_quote = cls._normalize_quote(quote)
        if not clean_quote:
            return False
        for post in source_posts:
            if clean_quote in cls._normalize_quote(post):
                return True
        return False

    @classmethod
    def find_evidence_post(
        cls, quote: Optional[str], posts: Iterable[object]
    ) -> Optional[object]:
        """Return the first post object containing the verbatim quote."""
        if not quote:
            return None
        clean_quote = cls._normalize_quote(quote)
        for post in posts:
            text = getattr(post, "text", "")
            if clean_quote in cls._normalize_quote(text):
                return post
        return None

    # ------------------------------------------------------------ consensus
    @staticmethod
    def _stance_value(stance: object) -> str:
        if isinstance(stance, Stance):
            return stance.value
        return str(stance).upper()

    @classmethod
    def resolve_platform_consensus(
        cls, platform_stances: Dict[str, object]
    ) -> Tuple[str, bool]:
        """
        Gate 8: never force an average when platforms disagree.

        Only meaningful BUY/SELL votes participate. WAIT is neutral; CONFLICT
        is treated as an explicit disagreement.
        """
        values = {cls._stance_value(value) for value in platform_stances.values()}
        if "CONFLICT" in values or ("BUY" in values and "SELL" in values):
            return "DIVERGENCE / CONFLICT", True
        if "BUY" in values:
            return "BULLISH", False
        if "SELL" in values:
            return "BEARISH", False
        return "NEUTRAL / WAIT", False

    @classmethod
    def needs_conflict_declaration(cls, sources: Iterable[object]) -> bool:
        stances = {
            getattr(source, "id", str(index)): getattr(source, "stance", Stance.WAIT)
            for index, source in enumerate(sources)
            if getattr(source, "status", None) is not None
        }
        _, conflict = cls.resolve_platform_consensus(stances)
        return conflict

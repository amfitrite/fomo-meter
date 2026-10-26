"""
Financial Relevance Gate (FRG).

Stage 1 rejects spam/gambling/non-financial noise.
Stage 2 requires a financial anchor, known trading slang, or finance emoji.
Stage 3 sanitizes prompt-injection style instructions before any classifier or
LLM sees the text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

from sentiment_llm.classifier import normalize_text

FINANCIAL_ANCHORS: Set[str] = {
    # instruments and pairs
    "btc", "btcusd", "eth", "ethusd", "xau", "xauusd", "gold", "silver",
    "eur", "eurusd", "usd", "usdollar", "dollar", "yen", "jpy", "gbp",
    "spy", "sp500", "ndx", "nasdaq", "dow", "dji", "vix", "oil", "wti",
    "brent", "commodity", "commodities", "forex", "fx", "crypto",
    "cryptocurrency", "altcoin", "bitcoin", "ethereum", "solana", "xrp",
    # market concepts
    "stock", "stocks", "share", "shares", "equity", "equities", "etf",
    "index", "bond", "bonds", "yield", "yields", "treasury", "debt",
    "market", "markets", "price", "prices", "chart", "candle", "volume",
    "liquidity", "resistance", "support", "breakout", "breakdown", "trend",
    "volatility", "rally", "selloff", "correction", "bull", "bear",
    # macro / catalysts
    "fed", "fomc", "cpi", "ppi", "inflation", "deflation", "recession",
    "rate", "rates", "interest", "economy", "economic", "gdp", "earnings",
    "revenue", "guidance", "ipo", "dividend", "buyback", "merger",
    "acquisition", "regulation", "regulator", "sec", "tariff", "sanction",
    "sanctions", "halving", "inflow", "inflows", "outflow", "outflows",
    "whale", "whales", "leverage", "liquidation", "liquidated", "long",
    "short", "position", "positions", "trade", "trading", "trader",
    "investor", "investment", "portfolio", "fund", "funds", "bank",
    "banks", "exchange", "mining", "halving", "airdrop", "defi", "nft",
}

PHRASE_ANCHORS = (
    "s&p", "s&p 500", "sp 500", "all-time high", "record high",
    "interest rate", "rate cut", "rate hike", "risk-on", "risk-off",
)

FINANCE_EMOJIS = ("🚀", "🌕", "📈", "📉", "🩸", "🐂", "🐻", "💎", "💀", "🦀")

# These are not "jailbreak detection" claims. They are explicit instruction
# patterns that should never be forwarded as if they were market evidence.
INJECTION_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+instructions?"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)forget\s+(all\s+)?(previous|prior|above)\s+instructions?"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)system\s*:\s*you\s+are"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)you\s+are\s+now\s+(a|an|the)\s+"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)return\s+(the\s+)?(json|stance|answer)\s*(with|as|=)?"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)output\s+stance\s*(=|:)?\s*(buy|sell|wait)"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)say\s+(buy|sell)\s+no\s+matter\s+what"), "[REDACTED_INJECTION]"),
    (re.compile(r"(?i)developer\s+mode|jailbreak|do\s+anything\s+now"), "[REDACTED_INJECTION]"),
)

BLACKLIST_REASON = "BLACKLIST_MATCH"
NO_CONTEXT_REASON = "NO_FINANCIAL_CONTEXT"


@dataclass
class RelevanceDecision:
    is_relevant: bool
    reason: str
    sanitized_text: Optional[str] = None
    injection_detected: bool = False
    matched_blacklist: Optional[str] = None

    @property
    def summary(self) -> str:
        if self.is_relevant:
            return "Lolos gate relevansi finansial."
        if self.reason.startswith(BLACKLIST_REASON):
            return "Ditolak: mengandung spam, judi, atau pola promosi terlarang."
        if self.reason.startswith(NO_CONTEXT_REASON):
            return "Ditolak: tidak ada konteks finansial yang terdeteksi."
        return self.reason


class FinancialRelevanceGate:
    """2-stage relevance gate with injection sanitization."""

    def __init__(self, data_dir: Optional[str | Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.blacklist_file = self.data_dir / "noise_blacklist.json"
        self.slang_kb_file = self.data_dir / "slang_knowledge_base.json"
        self.blacklist: Set[str] = set()
        self.slang_terms: Set[str] = set()
        self._slang_patterns: List[re.Pattern[str]] = []
        self.refresh()

    def refresh(self) -> None:
        self._load_blacklist()
        self._load_slang()
        self._compile_slang_patterns()

    def _load_blacklist(self) -> None:
        self.blacklist = set()
        try:
            if self.blacklist_file.exists():
                with self.blacklist_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.blacklist = {
                    str(item).strip().lower()
                    for item in payload.get("blacklist", [])
                    if str(item).strip()
                }
        except (OSError, ValueError):
            self.blacklist = set()

    def _load_slang(self) -> None:
        self.slang_terms = set()
        try:
            if self.slang_kb_file.exists():
                with self.slang_kb_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.slang_terms = {
                    str(term).strip().lower()
                    for term in payload.get("terms", {})
                    if str(term).strip()
                }
        except (OSError, ValueError):
            self.slang_terms = set()

    def _compile_slang_patterns(self) -> None:
        self._slang_patterns = []
        for term in sorted(self.slang_terms, key=len, reverse=True):
            escaped = re.escape(term)
            self._slang_patterns.append(re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE))

    # ------------------------------------------------------------------ core
    def evaluate(self, text: str) -> RelevanceDecision:
        original = normalize_text(text)
        if not original:
            return RelevanceDecision(False, "EMPTY_TEXT — teks kosong.")
        if len(original) > 4000:
            original = original[:4000]

        lower = original.lower()

        # Stage 1: noise / spam / gambling blacklist.
        for bad in self.blacklist:
            if bad and bad in lower:
                return RelevanceDecision(
                    False,
                    f"{BLACKLIST_REASON}: {bad} — konten judi/spam ditolak.",
                    original,
                    matched_blacklist=bad,
                )

        # Stage 2: financial context.
        tokens = set(re.findall(r"[a-z0-9&.$']+", lower))
        compact_tokens = {token.strip("$.") for token in tokens}
        has_anchor = bool(compact_tokens & FINANCIAL_ANCHORS)
        if not has_anchor:
            has_anchor = any(phrase in lower for phrase in PHRASE_ANCHORS)
        has_slang = any(pattern.search(lower) for pattern in self._slang_patterns)
        has_emoji = any(emoji in original for emoji in FINANCE_EMOJIS)

        if not (has_anchor or has_slang or has_emoji):
            return RelevanceDecision(
                False,
                f"{NO_CONTEXT_REASON} — tidak ada konteks finansial yang terdeteksi.",
                original,
            )

        # Stage 3: instruction sanitization.
        sanitized, injections = self._sanitize_prompt_injections(original)
        reason = "PASSED — lolos gate relevansi."
        if injections:
            reason = f"PASSED_WITH_SANITIZATION — {injections} instruksi mencurigakan diredaksi."
        return RelevanceDecision(True, reason, sanitized, injection_detected=bool(injections))

    @staticmethod
    def _sanitize_prompt_injections(text: str) -> Tuple[str, int]:
        sanitized = text
        count = 0
        for pattern, replacement in INJECTION_PATTERNS:
            sanitized, replacements = pattern.subn(replacement, sanitized)
            count += replacements
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized, count

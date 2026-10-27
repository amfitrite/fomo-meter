"""
Web-Grounded Autonomous Learning Engine for Sentiment SLM.

Adopts Google Search-Grounding / Retrieval-Augmented Continual Learning:
1. Detects unknown or low-confidence trading terms.
2. 3-Layer Trading-Only Firewall (strictly filters out non-trading topics).
3. Multi-Tier Resilient Search Fallback (Google News RSS -> Yahoo Finance -> Wikipedia -> DuckDuckGo).
4. Distills stance (BUY / SELL / WAIT) with exact verbatim quote and URL provenance.
5. Ingests into feedback.jsonl and learning_log.jsonl for background autonomous learning.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fomo_meter.config import (
    GOOGLE_NEWS_BASE_URL,
    LEARNING_LOG_FILE,
    NEWS_LOCALE,
    USER_AGENT,
)
from fomo_meter.relevance_gate import FinancialRelevanceGate
from fomo_meter.schema import Stance

LOGGER = logging.getLogger("fomo_meter.grounding")

# Trading indicator tokens required in retrieved snippets (Lapis 3 Firewall)
TRADING_INDICATOR_TOKENS = {
    "price", "prices", "market", "markets", "trade", "trading", "trader", "traders",
    "stock", "stocks", "crypto", "bitcoin", "ethereum", "rally", "dump", "bull", "bullish",
    "bear", "bearish", "liquidity", "support", "resistance", "breakout", "breakdown",
    "volume", "order", "orders", "investor", "investors", "shares", "loss", "profit",
    "whale", "whales", "short", "long", "leverage", "liquidated", "liquidation",
}

BULLISH_KEYWORDS = {
    "rally", "breakout", "bullish", "surge", "gain", "gains", "jump", "jumps",
    "buying", "accumulate", "accumulation", "bounce", "highs", "rebound", "inflow",
    "skyrocket", "outperform", "soar", "soars", "all-time high", "ath",
}

BEARISH_KEYWORDS = {
    "dump", "dumps", "dumping", "crash", "crashes", "crashing", "drop", "drops",
    "dropoff", "selloff", "bearish", "plunge", "plunges", "loss", "losses", "dive",
    "tumble", "tumbles", "fall", "falls", "liquidation", "liquidated", "breakdown",
    "rug pull", "collapse", "decline", "outflow", "cut loss", "bleed",
}


@dataclass
class GroundingResult:
    term: str
    stance: Stance
    confidence: float
    evidence_quote: str
    provenance_url: str
    source_engine: str
    raw_snippet: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["stance"] = self.stance.value
        return data


class TradingGrounder:
    """Multi-tier resilient search grounder strictly scoped to financial trading."""

    def __init__(
        self,
        relevance_gate: Optional[FinancialRelevanceGate] = None,
        log_file: Path = LEARNING_LOG_FILE,
    ) -> None:
        self.relevance_gate = relevance_gate or FinancialRelevanceGate()
        self.log_file = Path(log_file)
        self._lock = threading.RLock()
        self._cache: Dict[str, Optional[GroundingResult]] = {}

    # ----------------------------------------------------------- term detection
    def extract_candidate_terms(self, text: str) -> List[str]:
        """Extract candidate trading terms from text."""
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) >= 3]
        candidates: List[str] = []

        # 1-word candidate slangs
        for w in words:
            if w in {"rugpull", "depeg", "liquidated", "hodl", "fud", "retest", "pump", "dump"}:
                candidates.append(w)

        # 2-word candidate phrases
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            if any(k in pair for k in {"support", "resistance", "breakout", "breakdown", "rug pull", "dead cat", "liquidation cascade", "death cross", "golden cross", "bear trap", "bull trap"}):
                candidates.append(pair)

        return list(dict.fromkeys(candidates))

    # ----------------------------------------------------------------- search
    def ground_term(self, term: str) -> Optional[GroundingResult]:
        """Search and distill trading sentiment using Google-style multi-tier fallback."""
        term_clean = term.strip().lower()
        if not term_clean:
            return None

        with self._lock:
            if term_clean in self._cache:
                return self._cache[term_clean]

        # Tier 1: Google News RSS Syndication
        result = self._search_google_news(term_clean)

        # Tier 2: Yahoo Finance Search API
        if not result:
            result = self._search_yahoo_finance(term_clean)

        # Tier 3: Wikipedia / Financial Definition REST API
        if not result:
            result = self._search_wikipedia(term_clean)

        # Tier 4: DuckDuckGo Lite
        if not result:
            result = self._search_duckduckgo(term_clean)

        with self._lock:
            self._cache[term_clean] = result

        if result:
            self._log_learning_event(result)

        return result

    # ------------------------------------------------------------- Tier 1: Google
    def _search_google_news(self, term: str) -> Optional[GroundingResult]:
        try:
            query = f'"{term}" (trading OR crypto OR stock OR market)'
            params = dict(NEWS_LOCALE)
            params["q"] = query
            url = f"{GOOGLE_NEWS_BASE_URL}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            with urllib.request.urlopen(req, timeout=5.0) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            items = root.findall("./channel/item")
            for item in items[:3]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                description = (item.findtext("description") or "").strip()
                clean_desc = re.sub(r"<[^>]+>", " ", description)
                combined = f"{title}. {clean_desc}"

                # Lapis 3 Trading-Only Financial Filter
                if not self._is_financial_content(combined):
                    continue

                stance, conf = self._distill_stance(combined)
                return GroundingResult(
                    term=term,
                    stance=stance,
                    confidence=conf,
                    evidence_quote=title[:200],
                    provenance_url=link or url,
                    source_engine="Google News Syndication",
                    raw_snippet=combined[:300],
                )
        except Exception as exc:
            LOGGER.debug("Google News Grounding gagal untuk '%s': %s", term, exc)
        return None

    # ------------------------------------------------------ Tier 2: Yahoo Finance
    def _search_yahoo_finance(self, term: str) -> Optional[GroundingResult]:
        try:
            query = urllib.parse.quote(term)
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&newsCount=3"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            news_items = data.get("news") or []
            for item in news_items:
                title = str(item.get("title") or "").strip()
                link = str(item.get("link") or "").strip()
                publisher = str(item.get("publisher") or "Yahoo Finance").strip()
                combined = f"{title} ({publisher})"

                if not self._is_financial_content(combined):
                    continue

                stance, conf = self._distill_stance(combined)
                return GroundingResult(
                    term=term,
                    stance=stance,
                    confidence=conf,
                    evidence_quote=title[:200],
                    provenance_url=link or url,
                    source_engine="Yahoo Finance Search",
                    raw_snippet=combined[:300],
                )
        except Exception as exc:
            LOGGER.debug("Yahoo Finance Grounding gagal untuk '%s': %s", term, exc)
        return None

    # --------------------------------------------------------- Tier 3: Wikipedia
    def _search_wikipedia(self, term: str) -> Optional[GroundingResult]:
        try:
            slug = urllib.parse.quote(term.strip().replace(" ", "_").title())
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))

            title = str(data.get("title") or "")
            extract = str(data.get("extract") or "")
            page_url = str(data.get("content_urls", {}).get("desktop", {}).get("page") or url)
            combined = f"{title}: {extract}"

            if not self._is_financial_content(combined):
                return None

            stance, conf = self._distill_stance(combined)
            return GroundingResult(
                term=term,
                stance=stance,
                confidence=conf,
                evidence_quote=extract[:200] if extract else title,
                provenance_url=page_url,
                source_engine="Wikipedia Financial REST API",
                raw_snippet=combined[:300],
            )
        except Exception as exc:
            LOGGER.debug("Wikipedia Grounding gagal untuk '%s': %s", term, exc)
        return None

    # -------------------------------------------------------- Tier 4: DuckDuckGo
    def _search_duckduckgo(self, term: str) -> Optional[GroundingResult]:
        try:
            query = urllib.parse.quote(f"{term} finance trading")
            url = f"https://html.duckduckgo.com/html/?q={query}"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

            with urllib.request.urlopen(req, timeout=5.0) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Extract first snippet text
            snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
            links = re.findall(r'<a class="result__url[^>]*href="([^"]+)"', html)
            if snippets:
                clean_text = re.sub(r"<[^>]+>", " ", snippets[0]).strip()
                if self._is_financial_content(clean_text):
                    stance, conf = self._distill_stance(clean_text)
                    return GroundingResult(
                        term=term,
                        stance=stance,
                        confidence=conf,
                        evidence_quote=clean_text[:200],
                        provenance_url=links[0] if links else url,
                        source_engine="DuckDuckGo Lite",
                        raw_snippet=clean_text[:300],
                    )
        except Exception as exc:
            LOGGER.debug("DuckDuckGo Grounding gagal untuk '%s': %s", term, exc)
        return None

    # --------------------------------------------------------------- helpers
    def _is_financial_content(self, text: str) -> bool:
        lower = text.lower()
        count = sum(1 for tok in TRADING_INDICATOR_TOKENS if tok in lower)
        return count >= 1

    def _distill_stance(self, text: str) -> Tuple[Stance, float]:
        lower = text.lower()
        bull_score = sum(1.5 for k in BULLISH_KEYWORDS if k in lower)
        bear_score = sum(1.5 for k in BEARISH_KEYWORDS if k in lower)

        if bull_score > bear_score and bull_score >= 1.0:
            conf = min(0.98, 0.70 + (bull_score - bear_score) * 0.1)
            return Stance.BUY, round(conf, 2)
        elif bear_score > bull_score and bear_score >= 1.0:
            conf = min(0.98, 0.70 + (bear_score - bull_score) * 0.1)
            return Stance.SELL, round(conf, 2)
        return Stance.WAIT, 0.65

    def _log_learning_event(self, result: GroundingResult) -> None:
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            LOGGER.debug("Gagal mencatat learning log: %s", exc)

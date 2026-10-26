"""StockTwits public symbol stream adapter."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fomo_meter.config import REQUEST_TIMEOUT, TICKERS
from fomo_meter.schema import Post, SourceResult, SourceStatus, Stance
from fomo_meter.sources.analysis import aggregate_posts
from fomo_meter.sources.base import BaseSource, SourceFetchError


def _parse_created_at(value: Optional[str]) -> float:
    if not value:
        return time.time()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
        except (ValueError, OverflowError):
            return time.time()


def parse_stocktwits_messages(
    payload: Dict[str, Any],
    limit: int = 30,
    ticker_terms: Optional[List[str]] = None,
) -> List[Post]:
    """Convert the public API payload into Post models, preserving verbatim text."""
    posts: List[Post] = []
    seen: set[str] = set()
    terms = [term.lower() for term in (ticker_terms or []) if term]
    for message in payload.get("messages", []) or []:
        body = str(message.get("body") or "").strip()
        if not body:
            continue

        # Pure cashtag-only messages carry no analysable language.
        meaningful = re.sub(r"\$[A-Za-z0-9.]+", " ", body)
        meaningful = re.sub(r"https?://\S+", " ", meaningful)
        if len(re.findall(r"[A-Za-z]{2,}", meaningful)) == 0:
            continue

        if terms and not any(term in body.lower() for term in terms):
            continue

        normalized = re.sub(r"\s+", " ", body).lower()
        if normalized in seen:
            continue
        seen.add(normalized)

        user = message.get("user") or {}
        sentiment_block = ((message.get("entities") or {}).get("sentiment") or {})
        label = str(sentiment_block.get("basic") or "").lower()
        user_sentiment: Optional[Stance] = None
        if label == "bullish":
            user_sentiment = Stance.BUY
        elif label == "bearish":
            user_sentiment = Stance.SELL

        message_id = str(message.get("id") or f"st_{len(posts)}")
        posts.append(Post(
            id=message_id,
            platform="stocktwits",
            author=str(user.get("username") or "stocktwits_user"),
            text=body,
            created_at=_parse_created_at(message.get("created_at")),
            url=f"https://stocktwits.com/message/{message_id}",
            user_sentiment=user_sentiment,
        ))
        if len(posts) >= limit:
            break
    return posts


class StockTwitsSource(BaseSource):
    id = "stocktwits"
    name = "StockTwits"
    icon = "ST"
    kind = "Crowd"
    ttl = 90

    def _fetch(self, ticker: str) -> SourceResult:
        config = TICKERS.get(ticker.upper())
        if not config:
            raise SourceFetchError(f"Ticker {ticker} tidak dikenal.", SourceStatus.NO_DATA)
        symbol = config["stocktwits"]
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise SourceFetchError(
                "StockTwits membatasi permintaan (HTTP 429). Coba lagi beberapa saat.",
                SourceStatus.ERROR,
            )
        if response.status_code == 403:
            raise SourceFetchError(
                "StockTwits menolak akses publik (HTTP 403). Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )
        if response.status_code != 200:
            raise SourceFetchError(
                f"StockTwits mengembalikan HTTP {response.status_code}. Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceFetchError("Respons StockTwits bukan JSON yang valid.", SourceStatus.ERROR) from exc

        returned_symbol = str((payload.get("symbol") or {}).get("symbol") or "")
        symbol_verified = returned_symbol.upper() == str(symbol).upper()
        if not symbol_verified:
            return self._no_data_result(
                f"Simbol tidak cocok: diminta {symbol}, diterima {returned_symbol or 'kosong'}.",
                sample_count=0,
                provenance=url,
                symbol_verified=False,
            )

        posts = parse_stocktwits_messages(
            payload,
            ticker_terms=[symbol.lower(), *config.get("news_terms", [])],
        )
        if not posts:
            return self._no_data_result(
                f"Tidak ada pesan {symbol} yang relevan pada stream publik saat ini.",
                sample_count=0,
                provenance=url,
                symbol_verified=True,
            )
        result = aggregate_posts(
            posts,
            classifier=self._classifier,
            relevance_gate=self._gate,
            source_id=self.id,
            name=self.name,
            icon=self.icon,
            kind=self.kind,
            provenance=url,
            symbol_verified=True,
            min_analyzed=2,
            target_terms=[symbol.lower(), *config.get("news_terms", [])],
        )
        result.symbol_verified = symbol_verified
        return result

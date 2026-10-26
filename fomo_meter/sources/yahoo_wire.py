"""Yahoo Finance public RSS headline adapter.

Yahoo's per-symbol RSS feed is a useful second wire when Google News is
rate-limited. Headlines are real, dated, and linked. The adapter filters by
ticker terms and never invents a headline.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

from fomo_meter.config import REQUEST_TIMEOUT, TICKERS
from fomo_meter.schema import Post, SourceResult, SourceStatus
from fomo_meter.sources.analysis import aggregate_posts
from fomo_meter.sources.base import BaseSource, SourceFetchError


def build_yahoo_url(ticker: str) -> Optional[str]:
    config = TICKERS.get(ticker.upper())
    if not config:
        return None
    symbol = (config.get("market") or {}).get("yahoo")
    if not symbol:
        return None
    quoted = urllib.parse.quote(str(symbol), safe="")
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quoted}&region=US&lang=en-US"


def _parse_pub_date(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def parse_yahoo_rss(xml_text: str, ticker: str, max_age_hours: int = 72) -> List[Post]:
    config = TICKERS.get(ticker.upper())
    if not config:
        return []
    terms = [term.lower() for term in config.get("news_terms", [])]
    now = time.time()
    posts: List[Post] = []
    seen: set[str] = set()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    for item in root.findall("./channel/item"):
        title = re.sub(r"\s+", " ", (item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        if not title:
            continue
        published = _parse_pub_date(item.findtext("pubDate"))
        if published:
            age = now - published
            if age > max_age_hours * 3600:
                continue
            if age < -3600:
                published = now
        else:
            published = now
        lower = title.lower()
        if terms and not any(term in lower for term in terms):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", lower).strip()
        if key in seen:
            continue
        seen.add(key)
        post_id = hashlib.sha1(f"{title}|{link}".encode("utf-8")).hexdigest()[:16]
        posts.append(Post(
            id=f"yh_{post_id}",
            platform="yahoo",
            author="Yahoo Finance",
            text=title,
            created_at=float(published),
            url=link or None,
            source_name="Yahoo Finance",
        ))
        if len(posts) >= 20:
            break
    return posts


class YahooWireSource(BaseSource):
    id = "yahoo"
    name = "Yahoo Wire"
    icon = "YH"
    kind = "News"
    ttl = 300

    def _fetch(self, ticker: str) -> SourceResult:
        url = build_yahoo_url(ticker)
        config = TICKERS.get(ticker.upper())
        if not url or not config:
            raise SourceFetchError(f"Ticker {ticker} tidak dikenal.", SourceStatus.NO_DATA)
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise SourceFetchError(
                "Yahoo Finance membatasi permintaan (HTTP 429). Coba lagi nanti.",
                SourceStatus.ERROR,
            )
        if response.status_code != 200:
            raise SourceFetchError(
                f"Yahoo Wire mengembalikan HTTP {response.status_code}. Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )
        posts = parse_yahoo_rss(response.text, ticker)
        if not posts:
            return self._no_data_result(
                "Tidak ada headline Yahoo yang relevan dalam 72 jam terakhir.",
                sample_count=0,
                provenance=url,
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
            target_terms=config.get("news_terms", []),
        )
        result.symbol_verified = True
        return result

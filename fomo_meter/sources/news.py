"""Google News RSS adapter.

Headlines are real published articles. The adapter deliberately does not
invent summaries or snippets when Google returns nothing usable.
"""

from __future__ import annotations

import hashlib
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

from fomo_meter.config import NEWS_LOCALE, REQUEST_TIMEOUT, TICKERS
from fomo_meter.schema import Post, SourceResult, SourceStatus
from fomo_meter.sources.analysis import aggregate_posts
from fomo_meter.sources.base import BaseSource, SourceFetchError


def build_news_url(ticker: str) -> str:
    config = TICKERS.get(ticker.upper())
    query = (config or {}).get("news_query", ticker)
    params = {
        "q": f"{query} when:1d",
        "hl": NEWS_LOCALE["hl"],
        "gl": NEWS_LOCALE["gl"],
        "ceid": NEWS_LOCALE["ceid"],
    }
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(params)


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


def _clean_title(title: str, source: str) -> str:
    title = re.sub(r"\s+", " ", (title or "").strip())
    if source and title.lower().endswith(" - " + source.lower()):
        title = title[: -(len(source) + 3)].strip()
    return title


def parse_google_news(xml_text: str, ticker: str, max_age_hours: int = 48) -> List[Post]:
    """Parse RSS items, filter to ticker-relevant, fresh headlines."""
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
        raw_title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        source_name = item.findtext("source") or "Google News"
        published = _parse_pub_date(item.findtext("pubDate"))
        if not raw_title.strip():
            continue

        title = _clean_title(raw_title, source_name)
        if not title:
            continue
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

        normalized = re.sub(r"[^a-z0-9]+", " ", lower).strip()
        if normalized in seen:
            continue
        seen.add(normalized)

        post_id = hashlib.sha1(f"{title}|{link}".encode("utf-8")).hexdigest()[:16]
        posts.append(Post(
            id=f"news_{post_id}",
            platform="news",
            author=source_name,
            text=title,
            created_at=float(published),
            url=link or None,
            user_sentiment=None,
            source_name=source_name,
        ))
        if len(posts) >= 20:
            break
    return posts


class NewsSource(BaseSource):
    id = "news"
    name = "News Wire"
    icon = "NW"
    kind = "News"
    ttl = 300

    def _fetch(self, ticker: str) -> SourceResult:
        config = TICKERS.get(ticker.upper())
        if not config:
            raise SourceFetchError(f"Ticker {ticker} tidak dikenal.", SourceStatus.NO_DATA)
        url = build_news_url(ticker)

        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise SourceFetchError(
                "Google News membatasi permintaan (HTTP 429). Coba lagi nanti.",
                SourceStatus.ERROR,
            )
        if response.status_code != 200:
            raise SourceFetchError(
                f"News Wire mengembalikan HTTP {response.status_code}. Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )

        posts = parse_google_news(response.text, ticker)
        if not posts:
            return self._no_data_result(
                "Tidak ada headline baru yang relevan dalam 48 jam terakhir.",
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

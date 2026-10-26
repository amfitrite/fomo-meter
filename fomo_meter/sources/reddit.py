"""Reddit public listing adapter.

Reddit frequently rejects unauthenticated requests from datacenter/VPN IPs
with HTTP 403. When that happens the radar shows an explicit NO DATA state;
it never substitutes fabricated posts.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from fomo_meter.config import REQUEST_TIMEOUT, TICKERS
from fomo_meter.schema import Post, SourceResult, SourceStatus
from fomo_meter.sources.analysis import aggregate_posts
from fomo_meter.sources.base import BaseSource, SourceFetchError


def parse_reddit_listing(payload: Dict[str, Any], ticker: str) -> List[Post]:
    posts: List[Post] = []
    seen: set[str] = set()
    children = ((payload.get("data") or {}).get("children") or [])
    for child in children:
        data = child.get("data") or {}
        title = str(data.get("title") or "").strip()
        body = str(data.get("selftext") or "").strip()
        text = f"{title}. {body}".strip(". ").strip()
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        permalink = data.get("permalink")
        url = f"https://www.reddit.com{permalink}" if permalink else None
        posts.append(Post(
            id=str(data.get("id") or f"rd_{len(posts)}"),
            platform="reddit",
            author=str(data.get("author") or "reddit_user"),
            text=text[:1200],
            created_at=float(data.get("created_utc") or time.time()),
            url=url,
        ))
        if len(posts) >= 25:
            break
    return posts


class RedditSource(BaseSource):
    id = "reddit"
    name = "Reddit"
    icon = "RD"
    kind = "Forum"
    ttl = 300

    def _fetch(self, ticker: str) -> SourceResult:
        config = TICKERS.get(ticker.upper())
        if not config:
            raise SourceFetchError(f"Ticker {ticker} tidak dikenal.", SourceStatus.NO_DATA)
        subreddit = config.get("subreddit", "wallstreetbets")
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25&raw_json=1"

        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 403:
            raise SourceFetchError(
                "Reddit menolak akses publik dari jaringan ini (HTTP 403). "
                "Sumber ditandai NO DATA; tidak ada posting yang dipalsukan.",
                SourceStatus.ERROR,
            )
        if response.status_code == 429:
            raise SourceFetchError(
                "Reddit membatasi permintaan (HTTP 429). Coba lagi beberapa saat.",
                SourceStatus.ERROR,
            )
        if response.status_code != 200:
            raise SourceFetchError(
                f"Reddit mengembalikan HTTP {response.status_code}. Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceFetchError("Respons Reddit bukan JSON yang valid.", SourceStatus.ERROR) from exc

        posts = parse_reddit_listing(payload, ticker)
        if not posts:
            return self._no_data_result(
                f"r/{subreddit} tidak mengembalikan posting yang dapat dianalisis.",
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

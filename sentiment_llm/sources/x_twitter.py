"""Optional X / Twitter recent-search adapter.

No token means no data. The adapter never falls back to mock tweets.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

from sentiment_llm.config import REQUEST_TIMEOUT, TICKERS
from sentiment_llm.schema import Post, SourceResult, SourceStatus
from sentiment_llm.sources.analysis import aggregate_posts
from sentiment_llm.sources.base import BaseSource, SourceFetchError


CASHTAGS = {
    "BTCUSD": "$BTC",
    "ETHUSD": "$ETH",
    "XAUUSD": "$XAU",
    "SPY": "$SPY",
    "EURUSD": "$EURUSD",
}


def parse_tweets(payload: Dict[str, Any]) -> List[Post]:
    users = {
        str(user.get("id")): str(user.get("username") or "x_user")
        for user in ((payload.get("includes") or {}).get("users") or [])
    }
    posts: List[Post] = []
    for tweet in payload.get("data", []) or []:
        text = str(tweet.get("text") or "").strip()
        if not text:
            continue
        username = users.get(str(tweet.get("author_id")), "x_user")
        created = tweet.get("created_at")
        try:
            timestamp = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
        except (AttributeError, ValueError):
            timestamp = time.time()
        tweet_id = str(tweet.get("id") or f"x_{len(posts)}")
        posts.append(Post(
            id=tweet_id,
            platform="x",
            author=username,
            text=text,
            created_at=timestamp,
            url=f"https://x.com/{username}/status/{tweet_id}",
        ))
    return posts


class XSource(BaseSource):
    id = "x"
    name = "X / Twitter"
    icon = "X"
    kind = "Social"
    ttl = 120

    def _fetch(self, ticker: str) -> SourceResult:
        token = os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN")
        if not token:
            return SourceResult(
                id=self.id,
                name=self.name,
                icon=self.icon,
                kind=self.kind,
                status=SourceStatus.NEEDS_AUTH,
                reason=(
                    "Butuh X API Bearer Token. Set environment variable X_BEARER_TOKEN "
                    "lalu jalankan ulang aplikasi. Tidak ada tweet demo."
                ),
                stance=self._wait(),
                intensity=0,
                confidence=0.0,
                provenance="https://developer.x.com/",
                symbol_verified=False,
            )

        config = TICKERS.get(ticker.upper())
        if not config:
            raise SourceFetchError(f"Ticker {ticker} tidak dikenal.", SourceStatus.NO_DATA)
        cashtag = CASHTAGS.get(ticker.upper(), f"${config['short']}")
        query = f'({cashtag} OR "{config["label"]}") -is:retweet lang:en'
        url = "https://api.twitter.com/2/tweets/search/recent?" + urllib.parse.urlencode({
            "query": query,
            "max_results": 25,
            "tweet.fields": "created_at,public_metrics,lang",
            "expansions": "author_id",
            "user.fields": "username",
        })

        response = self.session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in (401, 403):
            raise SourceFetchError(
                "Token X ditolak (HTTP 401/403). Periksa X_BEARER_TOKEN dan tier API.",
                SourceStatus.NEEDS_AUTH,
            )
        if response.status_code == 429:
            raise SourceFetchError(
                "X API membatasi permintaan (HTTP 429). Coba lagi nanti.",
                SourceStatus.ERROR,
            )
        if response.status_code != 200:
            raise SourceFetchError(
                f"X API mengembalikan HTTP {response.status_code}. Tidak ada data yang dipalsukan.",
                SourceStatus.ERROR,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SourceFetchError("Respons X API bukan JSON yang valid.", SourceStatus.ERROR) from exc

        posts = parse_tweets(payload)
        if not posts:
            return self._no_data_result(
                "Tidak ada tweet publik yang cocok dalam jendela pencarian terbaru.",
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

    @staticmethod
    def _wait():
        from sentiment_llm.schema import Stance
        return Stance.WAIT

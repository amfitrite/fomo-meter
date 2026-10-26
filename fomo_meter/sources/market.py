"""Optional market context (price, 24h change, crypto fear & greed).

Market context is displayed separately and never feeds the sentiment score.
If it is unavailable, the UI shows a clear empty state.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
from typing import Any, Dict, Optional, Tuple

import requests

from fomo_meter.config import MARKET_CACHE_TTL, REQUEST_TIMEOUT, TICKERS, USER_AGENT
from fomo_meter.schema import MarketContext


class MarketSource:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        self._cache: Dict[str, Tuple[float, MarketContext]] = {}
        self._lock = threading.RLock()

    def fetch(self, ticker: str, force: bool = False) -> Optional[MarketContext]:
        key = ticker.upper()
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached and not force and (now - cached[0]) < MARKET_CACHE_TTL:
                return cached[1].model_copy(deep=True)

        config = TICKERS.get(key)
        if not config:
            return None
        market = config.get("market") or {}
        context = MarketContext(
            ticker=key,
            label=str(config.get("name") or key),
            fetched_at=time.time(),
        )

        try:
            price, change, source = self._fetch_price(market)
            if price is not None:
                context.price = float(price)
                context.change_pct = float(change) if change is not None else None
                context.source = source
            else:
                context.error = "Harga pasar tidak tersedia dari penyedia publik."
        except Exception as exc:  # Context is optional; never break the scan.
            context.error = f"Konteks pasar tidak tersedia ({type(exc).__name__})."

        if config.get("fear_greed"):
            try:
                fng = self._fetch_fear_greed()
                if fng:
                    context.fear_greed, context.fear_greed_label = fng
            except Exception:
                pass

        with self._lock:
            self._cache[key] = (time.time(), context.model_copy(deep=True))
        return context

    def _fetch_price(self, market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
        kind = market.get("kind")
        if kind == "coingecko":
            result = self._coingecko(market.get("id"))
            if result[0] is not None:
                return result
        yahoo = market.get("yahoo")
        if yahoo:
            return self._yahoo(yahoo)
        return None, None, "—"

    def _coingecko(self, coin_id: Optional[str]) -> Tuple[Optional[float], Optional[float], str]:
        if not coin_id:
            return None, None, "—"
        url = (
            "https://api.coingecko.com/api/v3/simple/price?"
            + urllib.parse.urlencode({
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            })
        )
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None, None, "—"
        payload = response.json()
        item = payload.get(coin_id) or {}
        price = item.get("usd")
        change = item.get("usd_24h_change")
        if price is None:
            return None, None, "—"
        return float(price), (float(change) if change is not None else None), "CoinGecko"

    def _yahoo(self, symbol: str) -> Tuple[Optional[float], Optional[float], str]:
        encoded = urllib.parse.quote(symbol, safe="")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            "?range=1d&interval=5m"
        )
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None, None, "—"
        payload = response.json()
        results = ((payload.get("chart") or {}).get("result") or [])
        if not results:
            return None, None, "—"
        meta = results[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        previous = meta.get("chartPreviousClose") or meta.get("previousClose")
        if price is None:
            return None, None, "—"
        change = None
        if previous:
            change = (float(price) - float(previous)) / float(previous) * 100.0
        return float(price), change, "Yahoo Finance"

    def _fetch_fear_greed(self) -> Optional[Tuple[int, str]]:
        url = "https://api.alternative.me/fng/?limit=1"
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None
        payload = response.json()
        data = payload.get("data") or []
        if not data:
            return None
        item = data[0]
        value = int(item.get("value"))
        label = str(item.get("value_classification") or "").strip() or "—"
        return value, label

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

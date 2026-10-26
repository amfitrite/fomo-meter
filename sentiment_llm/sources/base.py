"""Shared HTTP/caching primitives for public data sources."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import requests

from sentiment_llm.config import REQUEST_TIMEOUT, USER_AGENT
from sentiment_llm.schema import SourceResult, SourceStatus, Stance


class SourceFetchError(Exception):
    def __init__(self, reason: str, status: SourceStatus = SourceStatus.ERROR):
        super().__init__(reason)
        self.reason = reason
        self.status = status


class BaseSource(ABC):
    id: str = "base"
    name: str = "Base"
    icon: str = "?"
    kind: str = "Source"
    ttl: int = 90
    requires_auth: bool = False

    def __init__(self, classifier=None, relevance_gate=None) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/xml, application/xml, text/html;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,id;q=0.7",
        })
        self._cache: Dict[str, Tuple[float, SourceResult]] = {}
        self._cache_lock = threading.RLock()
        self._classifier = classifier
        self._gate = relevance_gate

    def attach_nlp(self, classifier, relevance_gate) -> None:
        self._classifier = classifier
        self._gate = relevance_gate

    # ------------------------------------------------------------------ cache
    def fetch(self, ticker: str, force: bool = False) -> SourceResult:
        key = ticker.upper()
        now = time.time()
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and not force:
                cached_at, result = cached
                ttl = 30 if result.status in (SourceStatus.ERROR, SourceStatus.NEEDS_AUTH) else self.ttl
                if (now - cached_at) < ttl:
                    copy = result.model_copy(deep=True)
                    copy.cached = True
                    return copy

        started = time.perf_counter()
        try:
            result = self._fetch(key)
        except SourceFetchError as exc:
            result = self._error_result(exc.reason, exc.status)
        except requests.Timeout:
            result = self._error_result(
                f"Sumber timeout setelah {REQUEST_TIMEOUT} detik. Data tidak diisi palsu."
            )
        except requests.RequestException as exc:
            result = self._error_result(
                f"Koneksi ke sumber gagal: {type(exc).__name__}. Data tidak diisi palsu."
            )
        except Exception as exc:  # Defensive: a broken adapter must not crash the radar.
            result = self._error_result(
                f"Adapter gagal memproses data: {type(exc).__name__}. Data tidak diisi palsu."
            )

        result.fetched_at = time.time()
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        result.cached = False
        with self._cache_lock:
            self._cache[key] = (time.time(), result.model_copy(deep=True))
        return result

    # --------------------------------------------------------------- helpers
    def _error_result(self, reason: str, status: SourceStatus = SourceStatus.ERROR) -> SourceResult:
        return SourceResult(
            id=self.id,
            name=self.name,
            icon=self.icon,
            kind=self.kind,
            status=status,
            reason=reason,
            stance=Stance.WAIT,
            intensity=0,
            confidence=0.0,
            provenance=self._provenance_hint(),
            symbol_verified=False,
        )

    def _no_data_result(
        self,
        reason: str,
        sample_count: int = 0,
        provenance: Optional[str] = None,
        symbol_verified: bool = True,
    ) -> SourceResult:
        return SourceResult(
            id=self.id,
            name=self.name,
            icon=self.icon,
            kind=self.kind,
            status=SourceStatus.NO_DATA,
            reason=reason,
            stance=Stance.WAIT,
            intensity=0,
            confidence=0.0,
            sample_count=sample_count,
            provenance=provenance or self._provenance_hint(),
            symbol_verified=symbol_verified,
        )

    def _provenance_hint(self) -> Optional[str]:
        return None

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    @abstractmethod
    def _fetch(self, ticker: str) -> SourceResult:
        raise NotImplementedError

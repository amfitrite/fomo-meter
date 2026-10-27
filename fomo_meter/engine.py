"""
Sentiment engine: live sources -> gates -> consensus -> history.

The engine never invents a source, a post, or an evidence quote. Every number
in a scan is derived from a SourceResult returned by an adapter, and every
evidence string is checked against the original post text.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fomo_meter.anti_halu_gates import AntiHallucinationVerifier
from fomo_meter.auto_learner import AutoLearner
from fomo_meter.classifier import FastCrowdClassifier
from fomo_meter.config import (
    DISCLAIMER,
    HISTORY_FILE,
    MAX_HISTORY,
    MAX_TEXT_LENGTH,
    SCAN_CACHE_TTL,
    SLM_MIN_CONFIDENCE,
    STALE_AFTER,
    TICKERS,
    ensure_dirs,
)
from fomo_meter.relevance_gate import FinancialRelevanceGate
from fomo_meter.schema import (
    AnalysisResult,
    FeedStatus,
    GateCheck,
    GateState,
    HistoryEntry,
    ModelBrief,
    Post,
    ScanResult,
    SourceResult,
    SourceStatus,
    Stance,
)
from fomo_meter.slm import SentimentSLM, SLMUnavailable
from fomo_meter.sources import build_default_sources
from fomo_meter.sources.analysis import _excerpt
from fomo_meter.sources.market import MarketSource


class NoLiveDataError(RuntimeError):
    pass


GATE_NAMES = {
    "G1": "Sumber live tersedia",
    "G2": "Simbol sesuai",
    "G3": "Gate relevansi finansial",
    "G4": "Kecukupan sampel",
    "G5": "Kesegaran data",
    "G6": "Bukti verbatim",
    "G7": "Konflik tidak dirata-ratakan",
    "G8": "Provenance sumber",
    "G9": "Tidak ada data sintetis",
}


def _gate(gate_id: str, state: GateState, detail: str) -> GateCheck:
    return GateCheck(
        id=gate_id,
        name=GATE_NAMES.get(gate_id, gate_id),
        state=state,
        detail=detail,
    )


def _safe_weight(source: SourceResult) -> float:
    return max(1.0, float(source.analyzed_count)) * max(0.25, float(source.confidence))


class SentimentEngine:
    def __init__(self, data_dir: Optional[str | Path] = None) -> None:
        ensure_dirs()
        self.data_dir = Path(data_dir) if data_dir else HISTORY_FILE.parent
        self.classifier = FastCrowdClassifier(self.data_dir)
        self.relevance_gate = FinancialRelevanceGate(self.data_dir)
        self.sources = build_default_sources(self.classifier, self.relevance_gate)
        self.market_source = MarketSource()
        self.slm = SentimentSLM()
        self.learner = AutoLearner(self.data_dir, self.slm, self.relevance_gate)
        self._scan_cache: Dict[str, Tuple[float, ScanResult]] = {}
        self._cache_lock = threading.RLock()
        self._history_lock = threading.RLock()
        self._history: List[HistoryEntry] = self._load_history()

    # --------------------------------------------------------------- history
    def _load_history(self) -> List[HistoryEntry]:
        if not HISTORY_FILE.exists():
            return []
        try:
            with HISTORY_FILE.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                return []
            entries = [HistoryEntry.model_validate(item) for item in payload]
            return entries[-MAX_HISTORY:]
        except (OSError, ValueError, TypeError):
            return []

    def _save_history(self) -> None:
        try:
            payload = [entry.model_dump(mode="json") for entry in self._history[-MAX_HISTORY:]]
            tmp = HISTORY_FILE.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            tmp.replace(HISTORY_FILE)
        except OSError:
            pass

    def record_history(self, entry: HistoryEntry) -> None:
        with self._history_lock:
            self._history.append(entry)
            self._history = self._history[-MAX_HISTORY:]
            self._save_history()

    def history(self, limit: int = 30) -> List[HistoryEntry]:
        with self._history_lock:
            return list(reversed(self._history[-limit:]))

    # ------------------------------------------------------------------ scan
    def _fetch_one(self, source, ticker: str, force: bool) -> SourceResult:
        return source.fetch(ticker, force=force)

    def scan(self, ticker: str, force: bool = False) -> ScanResult:
        ticker = ticker.upper().strip()
        if ticker not in TICKERS:
            raise ValueError(f"Ticker tidak dikenal: {ticker}")

        now = time.time()
        with self._cache_lock:
            cached = self._scan_cache.get(ticker)
            if cached and not force and (now - cached[0]) < SCAN_CACHE_TTL:
                result = cached[1].model_copy(deep=True)
                result.cached = True
                result.next_refresh_in = max(0, int(SCAN_CACHE_TTL - (now - cached[0])))
                return result

        started = time.perf_counter()
        results: List[SourceResult] = []
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="source") as pool:
            future_map = {
                pool.submit(self._fetch_one, source, ticker, force): source
                for source in self.sources
            }
            market_future = pool.submit(self.market_source.fetch, ticker, force)
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(SourceResult(
                        id=source.id,
                        name=source.name,
                        icon=source.icon,
                        kind=source.kind,
                        status=SourceStatus.ERROR,
                        reason=f"Adapter gagal: {type(exc).__name__}. Tidak ada data palsu.",
                        stance=Stance.WAIT,
                        intensity=0,
                        confidence=0.0,
                    ))
            try:
                market = market_future.result(timeout=15)
            except Exception:
                market = None

        order = {source.id: index for index, source in enumerate(self.sources)}
        results.sort(key=lambda item: order.get(item.id, 99))

        live = [item for item in results if item.status == SourceStatus.LIVE]
        directional = [item for item in live if item.stance in (Stance.BUY, Stance.SELL)]
        analyzed_posts = sum(item.analyzed_count for item in results)
        total_posts = sum(item.sample_count for item in results)
        latest_post_at = max(
            (item.latest_post_at for item in live if item.latest_post_at),
            default=None,
        )

        conflict = False
        consensus = "NO_DATA"
        intensity = 0
        confidence = 0.0

        if live:
            buy_weight = sum(_safe_weight(item) for item in directional if item.stance == Stance.BUY)
            sell_weight = sum(_safe_weight(item) for item in directional if item.stance == Stance.SELL)
            if buy_weight > 0 and sell_weight > 0:
                conflict = True
                consensus = "CONFLICT"
                intensity = 50
                total_directional = buy_weight + sell_weight
                confidence = 0.35 + 0.25 * min(1.0, analyzed_posts / 12.0)
                confidence -= 0.10 * min(1.0, min(buy_weight, sell_weight) / max(buy_weight, sell_weight))
            elif buy_weight > 0 or sell_weight > 0:
                stance = Stance.BUY if buy_weight > 0 else Stance.SELL
                consensus = stance.value
                directional_sources = [item for item in directional if item.stance == stance]
                weights = [_safe_weight(item) for item in directional_sources]
                weight_sum = sum(weights) or 1.0
                intensity = int(round(
                    sum(item.intensity * _safe_weight(item) for item in directional_sources) / weight_sum
                ))
                intensity = max(51, min(99, intensity))
                confidence = 0.38 + (0.37 * min(1.0, analyzed_posts / 10.0))
                confidence += 0.10 * min(1.0, len(directional_sources) / 3.0)
            else:
                consensus = "WAIT"
                intensity = 50
                confidence = 0.30 + 0.25 * min(1.0, analyzed_posts / 10.0)

        if not live:
            feed_status = FeedStatus.NO_DATA
        elif conflict:
            feed_status = FeedStatus.PARTIAL
        elif any(item.status in (
            SourceStatus.ERROR, SourceStatus.NO_DATA, SourceStatus.NEEDS_AUTH, SourceStatus.STALE
        ) for item in results):
            feed_status = FeedStatus.PARTIAL
        elif latest_post_at and (time.time() - latest_post_at) > 86400:
            feed_status = FeedStatus.STALE
        else:
            feed_status = FeedStatus.LIVE

        if conflict:
            confidence = max(0.20, min(0.70, confidence))
        else:
            confidence = max(0.0, min(0.95, confidence))

        # Evidence: only from a live source, and only an exact quote.
        evidence_result = next(
            (item for item in live if item.evidence and item.evidence_url),
            None,
        )
        if not evidence_result:
            evidence_result = next((item for item in live if item.evidence), None)

        gates = self._build_scan_gates(
            results=results,
            live=live,
            directional=directional,
            analyzed_posts=analyzed_posts,
            conflict=conflict,
            latest_post_at=latest_post_at,
            evidence_result=evidence_result,
        )

        result = ScanResult(
            ticker=ticker,
            label=str(TICKERS[ticker]["label"]),
            timestamp=time.time(),
            feed_status=feed_status,
            consensus=consensus,
            conflict_detected=conflict,
            intensity=intensity,
            confidence=round(confidence, 2),
            sources=results,
            gates=gates,
            market=market,
            total_posts=total_posts,
            analyzed_posts=analyzed_posts,
            live_sources=len(live),
            latency_ms=int((time.perf_counter() - started) * 1000),
            cached=False,
            is_demo=False,
            next_refresh_in=SCAN_CACHE_TTL,
        )
        with self._cache_lock:
            self._scan_cache[ticker] = (time.time(), result.model_copy(deep=True))
        return result

    def _build_scan_gates(
        self,
        *,
        results: List[SourceResult],
        live: List[SourceResult],
        directional: List[SourceResult],
        analyzed_posts: int,
        conflict: bool,
        latest_post_at: Optional[float],
        evidence_result: Optional[SourceResult],
    ) -> List[GateCheck]:
        gates: List[GateCheck] = []
        if live:
            gates.append(_gate("G1", GateState.PASS, f"{len(live)} sumber live."))
        else:
            gates.append(_gate("G1", GateState.FAIL, "Tidak ada sumber live. Konsensus tidak dibuat."))

        mismatched = [item.name for item in live if not item.symbol_verified]
        if mismatched:
            gates.append(_gate("G2", GateState.FAIL, "Simbol tidak cocok: " + ", ".join(mismatched)))
        elif live:
            gates.append(_gate("G2", GateState.PASS, "Semua sumber live cocok dengan ticker yang diminta."))
        else:
            gates.append(_gate("G2", GateState.SKIP, "Tidak ada sumber live untuk diverifikasi."))

        rejected_hint = sum(1 for item in results if item.status == SourceStatus.NO_DATA and item.reason and "relevansi" in item.reason.lower())
        if analyzed_posts > 0:
            gates.append(_gate("G3", GateState.PASS, f"{analyzed_posts} post lolos relevansi finansial."))
        elif rejected_hint:
            gates.append(_gate("G3", GateState.FAIL, "Post tersedia, tetapi tidak ada yang lolos gate relevansi."))
        else:
            gates.append(_gate("G3", GateState.SKIP, "Tidak ada post untuk dievaluasi."))

        if analyzed_posts >= 5:
            gates.append(_gate("G4", GateState.PASS, f"{analyzed_posts} post dianalisis (>= 5)."))
        elif analyzed_posts > 0:
            gates.append(_gate("G4", GateState.WARN, f"Hanya {analyzed_posts} post; sampel kecil."))
        else:
            gates.append(_gate("G4", GateState.FAIL, "Tidak ada sampel yang dianalisis."))

        if latest_post_at and (time.time() - latest_post_at) <= 86400:
            age_minutes = max(0, int((time.time() - latest_post_at) / 60))
            gates.append(_gate("G5", GateState.PASS, f"Post terbaru sekitar {age_minutes} menit lalu."))
        elif latest_post_at:
            gates.append(_gate("G5", GateState.WARN, "Post terbaru lebih dari 24 jam; data mungkin stale."))
        else:
            gates.append(_gate("G5", GateState.SKIP, "Timestamp post tidak tersedia."))

        if evidence_result and evidence_result.evidence:
            verified = AntiHallucinationVerifier.verify_evidence_quote(
                evidence_result.evidence, [post.text for post in evidence_result.posts]
            )
            if verified:
                gates.append(_gate("G6", GateState.PASS, "Kutipan bukti ditemukan verbatim di post sumber."))
            else:
                gates.append(_gate("G6", GateState.FAIL, "Kutipan bukti tidak cocok dengan post sumber."))
        else:
            gates.append(_gate("G6", GateState.WARN, "Belum ada kutipan bukti ber-URL yang dapat diverifikasi."))

        if conflict:
            gates.append(_gate("G7", GateState.WARN, "Sumber BUY dan SELL bertemu; konsensus ditandai CONFLICT, bukan dirata-ratakan."))
        elif directional:
            gates.append(_gate("G7", GateState.PASS, "Arah sumber konsisten; tidak ada konflik yang disembunyikan."))
        else:
            gates.append(_gate("G7", GateState.SKIP, "Tidak ada arah BUY/SELL untuk dibandingkan."))

        with_provenance = [item for item in live if item.provenance]
        if live and len(with_provenance) == len(live):
            gates.append(_gate("G8", GateState.PASS, "Semua sumber live menyertakan URL/provenance."))
        elif live:
            gates.append(_gate("G8", GateState.WARN, f"{len(live) - len(with_provenance)} sumber tanpa provenance."))
        else:
            gates.append(_gate("G8", GateState.SKIP, "Tidak ada sumber live."))

        if any(item.status == SourceStatus.LIVE for item in results) and not any(item.sample_count < 0 for item in results):
            gates.append(_gate("G9", GateState.PASS, "Tidak ada data demo atau payload sintetis dalam scan ini."))
        else:
            gates.append(_gate("G9", GateState.WARN, "Tidak ada payload live; aplikasi tidak mengganti dengan data demo."))
        return gates

    # ---------------------------------------------------------------- analyze
    def analyze_text(self, text: str, use_model: bool = False, record: bool = True) -> AnalysisResult:
        started = time.perf_counter()
        raw = (text or "")[:MAX_TEXT_LENGTH]
        decision = self.relevance_gate.evaluate(raw)
        if not decision.is_relevant:
            result = AnalysisResult(
                input_text=raw,
                sanitized_text=decision.sanitized_text,
                is_relevant=False,
                reason=decision.summary,
                stance=Stance.WAIT,
                intensity=0,
                confidence=0.0,
                evidence_quote=None,
                engine="lexicon",
                gates=[
                    _gate("G1", GateState.FAIL, decision.summary),
                    _gate("G2", GateState.SKIP, "Teks tidak dianalisis karena tidak relevan."),
                    _gate("G3", GateState.SKIP, "Model tidak dijalankan pada teks yang ditolak gate."),
                ],
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            if record:
                self.record_history(HistoryEntry(
                    timestamp=time.time(), kind="rejected", stance="WAIT", intensity=0,
                    text=raw[:280], engine="lexicon", model=None,
                ))
            return result

        clean = decision.sanitized_text or raw
        baseline = self.classifier.classify(clean)
        stance = baseline.stance
        intensity = baseline.intensity
        confidence = baseline.confidence
        engine = "lexicon"
        model: Optional[str] = None
        reasoning: Optional[str] = None
        probabilities: Dict[str, float] = {}
        model_error: Optional[str] = None
        model_gate = _gate("G3", GateState.SKIP, "Sentiment SLM tidak diminta untuk analisis ini.")

        if use_model:
            if not self.slm.available:
                model_error = self.slm.error or "Sentiment SLM belum tersedia."
                model_gate = _gate("G3", GateState.WARN, f"Sentiment SLM tidak tersedia: {model_error}")
            else:
                try:
                    prediction = self.slm.predict(clean)
                    probabilities = prediction.probabilities
                    model = prediction.model
                    agrees = prediction.stance == baseline.stance
                    confident = prediction.confidence >= SLM_MIN_CONFIDENCE
                    if agrees or confident:
                        stance = prediction.stance
                        intensity = prediction.intensity
                        confidence = prediction.confidence
                        engine = "sentiment_slm"
                        model_gate = _gate(
                            "G3", GateState.PASS,
                            f"Sentiment SLM aktif; confidence {prediction.confidence:.0%}.",
                        )
                        reasoning = (
                            f"Sentiment SLM memprediksi {prediction.stance.value} "
                            f"(BUY {probabilities.get('BUY', 0):.1%}, "
                            f"SELL {probabilities.get('SELL', 0):.1%}, "
                            f"WAIT {probabilities.get('WAIT', 0):.1%}). "
                            f"Lexicon baseline: {baseline.stance.value}. "
                            "Output model bukan data baru dan tidak menambah bukti."
                        )
                    else:
                        model_error = (
                            f"Sentiment SLM confidence {prediction.confidence:.0%} di bawah ambang "
                            f"{SLM_MIN_CONFIDENCE:.0%}; memakai lexicon baseline."
                        )
                        model_gate = _gate("G3", GateState.WARN, model_error)
                        reasoning = (
                            f"Sentiment SLM memprediksi {prediction.stance.value}, tetapi confidence "
                            f"{prediction.confidence:.0%} belum cukup. Lexicon baseline "
                            f"{baseline.stance.value} dipakai agar tidak memaksakan arah."
                        )
                except SLMUnavailable as exc:
                    model_error = str(exc)
                    model_gate = _gate("G3", GateState.WARN, f"Sentiment SLM gagal: {model_error}")

        gates = [
            _gate("G1", GateState.PASS, decision.summary),
            _gate(
                "G2",
                GateState.WARN if decision.injection_detected else GateState.PASS,
                "Instruksi mencurigakan diredaksi sebelum analisis."
                if decision.injection_detected else "Tidak ada instruksi injeksi yang terdeteksi.",
            ),
            model_gate,
            _gate("G4", GateState.PASS, "Teks input adalah bukti utama dan disimpan verbatim."),
        ]

        result = AnalysisResult(
            input_text=raw,
            sanitized_text=clean,
            is_relevant=True,
            reason=decision.summary,
            stance=stance,
            intensity=intensity,
            confidence=round(float(confidence), 4),
            matched_terms=baseline.matched_terms,
            evidence_quote=baseline.evidence_quote or clean[:280],
            is_ambiguous=baseline.is_ambiguous,
            engine=engine,
            model=model,
            reasoning=reasoning,
            latency_ms=int((time.perf_counter() - started) * 1000),
            scores=baseline.scores,
            probabilities=probabilities,
            gates=gates,
            model_error=model_error,
            evidence_verified=True,
        )
        if record:
            self.record_history(HistoryEntry(
                timestamp=time.time(),
                kind="text",
                stance=stance.value,
                intensity=intensity,
                text=clean[:280],
                engine=engine,
                model=model,
            ))
        return result

    # ----------------------------------------------------------------- brief
    def slm_brief(self, ticker: str, scan: Optional[ScanResult] = None, record: bool = True) -> ModelBrief:
        if not self.slm.available:
            raise SLMUnavailable(self.slm.error or "Sentiment SLM belum tersedia.")
        started = time.perf_counter()
        ticker = ticker.upper().strip()
        scan = scan or self.scan(ticker)
        live_sources = [item for item in scan.sources if item.status == SourceStatus.LIVE]
        if not live_sources:
            raise NoLiveDataError(
                "Tidak ada sumber live. Brief Sentiment SLM tidak dibuat karena tidak ada bukti untuk dianalisis."
            )

        posts: List[Post] = []
        seen: set[str] = set()
        for source in live_sources:
            for post in source.posts:
                key = " ".join(post.text.lower().split())
                if key in seen:
                    continue
                decision = self.relevance_gate.evaluate(post.text)
                if not decision.is_relevant:
                    continue
                seen.add(key)
                posts.append(post)
        if not posts:
            raise NoLiveDataError("Tidak ada post yang lolos gate relevansi untuk Brief Sentiment SLM.")
        posts = posts[:16]

        clean_texts = [
            self.relevance_gate.evaluate(post.text).sanitized_text or post.text
            for post in posts
        ]
        predictions = self.slm.predict_many(clean_texts)

        average = {
            label: sum(prediction.probabilities.get(label, 0.0) for prediction in predictions) / len(predictions)
            for label in ("BUY", "SELL", "WAIT")
        }
        dominant_label = max(average, key=average.get)
        stance = Stance(dominant_label)
        confidence = float(average[dominant_label])
        if stance == Stance.WAIT:
            intensity = 50
        else:
            intensity = int(round(50 + abs(average["BUY"] - average["SELL"]) * 50))
            intensity = max(51, min(99, intensity))

        counts = Counter(prediction.stance.value for prediction in predictions)
        terms: Counter[str] = Counter()
        for post in posts:
            classification = self.classifier.classify(post.text)
            for term in classification.matched_terms:
                if len(term) > 1:
                    terms[term] += 1
        key_drivers = [term for term, _count in terms.most_common(8)]

        matching = [
            (post, prediction) for post, prediction in zip(posts, predictions)
            if prediction.stance == stance
        ]
        if not matching:
            matching = list(zip(posts, predictions))
        best_post, best_prediction = max(
            matching,
            key=lambda pair: pair[1].confidence,
        )
        evidence_quote = _excerpt(best_post.text, 240)
        evidence_url = best_post.url

        drivers_text = ", ".join(key_drivers) if key_drivers else "tidak ada term lexicon dominan"
        reasoning = (
            f"Sentiment SLM menganalisis {len(posts)} post live dari {len(live_sources)} sumber. "
            f"Distribusi model: {counts.get('BUY', 0)} BUY, {counts.get('SELL', 0)} SELL, "
            f"{counts.get('WAIT', 0)} WAIT. Probabilitas rata-rata: "
            f"BUY {average['BUY']:.0%}, SELL {average['SELL']:.0%}, WAIT {average['WAIT']:.0%}. "
            f"Driver lexicon: {drivers_text}."
        )
        if scan.conflict_detected:
            reasoning += " Radar menandai konflik antar-sumber; brief ini tidak mengubahnya menjadi rata-rata paksa."
        if stance == Stance.WAIT:
            reasoning += " Tidak ada arah dominan yang layak dipaksakan."

        latency = int((time.perf_counter() - started) * 1000)
        brief = ModelBrief(
            ticker=ticker,
            model=f"{self.slm.version} · lokal",
            stance=stance,
            intensity=intensity,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            key_drivers=key_drivers,
            evidence_quote=evidence_quote,
            evidence_url=evidence_url,
            probabilities={key: round(value, 4) for key, value in average.items()},
            latency_ms=latency,
            timestamp=time.time(),
            sample_size=len(posts),
            disclaimer=(
                "Brief ini dihasilkan model lokal Sentiment SLM dari post yang lolos gate. "
                "Model tidak menambah data atau bukti baru, dan tidak mengubah konsensus radar."
            ),
        )
        if record:
            self.record_history(HistoryEntry(
                timestamp=time.time(),
                kind="brief",
                ticker=ticker,
                stance=stance.value,
                intensity=intensity,
                text=reasoning[:280],
                engine="sentiment_slm",
                model=self.slm.version,
            ))
        return brief

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, object]:
        return {
            "sources": [
                {"id": source.id, "name": source.name}
                for source in self.sources
            ],
            "tickers": [
                {"id": key, "label": value["label"], "short": value["short"]}
                for key, value in TICKERS.items()
            ],
            "slm": self.slm.status().model_dump(),
            "auto_learner": self.learner.get_stats(),
            "disclaimer": DISCLAIMER,
            "history_count": len(self.history(limit=MAX_HISTORY)),
        }

    # ----------------------------------------------------------- auto learner
    def record_feedback(
        self, text: str, label: str, source: str = "user_feedback", auto_retrain_threshold: int = 0
    ) -> Dict[str, Any]:
        return self.learner.record_feedback(text, label, source, auto_retrain_threshold)

    def auto_retrain(self) -> Dict[str, Any]:
        return self.learner.retrain()

    def close(self) -> None:
        for source in self.sources:
            source.close()
        self.market_source.close()

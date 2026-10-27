from __future__ import annotations

import tempfile
import time
import unittest

from fomo_meter.engine import SentimentEngine
from fomo_meter.schema import (
    MarketContext,
    Post,
    SourceResult,
    SourceStatus,
    Stance,
)


class FakeSource:
    def __init__(self, result: SourceResult) -> None:
        self.result = result
        self.id = result.id
        self.name = result.name
        self.icon = result.icon
        self.kind = result.kind
        self.closed = False

    def fetch(self, ticker: str, force: bool = False) -> SourceResult:
        return self.result.model_copy(deep=True)

    def close(self) -> None:
        self.closed = True


class FakeMarket:
    def fetch(self, ticker: str, force: bool = False) -> MarketContext:
        return MarketContext(
            ticker=ticker,
            label="Test",
            price=100.0,
            change_pct=1.5,
            source="Test market",
            fetched_at=time.time(),
        )

    def close(self) -> None:
        pass


def live_result(source_id: str, stance: Stance, text: str) -> SourceResult:
    now = time.time()
    return SourceResult(
        id=source_id,
        name=source_id.title(),
        icon=source_id[:2].upper(),
        kind="Test",
        status=SourceStatus.LIVE,
        stance=stance,
        intensity=80,
        confidence=0.8,
        sample_count=6,
        analyzed_count=6,
        bullish_count=6 if stance == Stance.BUY else 0,
        bearish_count=6 if stance == Stance.SELL else 0,
        neutral_count=0,
        evidence=text,
        evidence_url=f"https://example.com/{source_id}",
        latest_post_at=now,
        fetched_at=now,
        provenance=f"https://example.com/{source_id}",
        posts=[Post(
            id=f"{source_id}-1",
            platform="test",
            author="tester",
            text=text,
            created_at=now,
            url=f"https://example.com/{source_id}",
        )],
    )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.engine = SentimentEngine(data_dir=self.tempdir.name)
        self.engine.market_source = FakeMarket()

    def tearDown(self) -> None:
        self.engine.close()
        self.tempdir.cleanup()

    def test_conflict_is_declared_not_averaged(self) -> None:
        buy = live_result("buy", Stance.BUY, "Bitcoin breakout with volume")
        sell = live_result("sell", Stance.SELL, "Bitcoin breakdown, cut loss")
        self.engine.sources = [FakeSource(buy), FakeSource(sell)]
        scan = self.engine.scan("BTCUSD", force=True)
        self.assertTrue(scan.conflict_detected)
        self.assertEqual(scan.consensus, "CONFLICT")
        self.assertEqual(scan.feed_status.value, "PARTIAL")
        self.assertEqual(scan.live_sources, 2)
        gate = next(item for item in scan.gates if item.id == "G7")
        self.assertEqual(gate.state.value, "WARN")

    def test_no_live_source_gives_no_data(self) -> None:
        now = time.time()
        error = SourceResult(
            id="broken",
            name="Broken",
            icon="BR",
            kind="Test",
            status=SourceStatus.ERROR,
            reason="HTTP 403",
            stance=Stance.WAIT,
            latest_post_at=None,
            fetched_at=now,
        )
        self.engine.sources = [FakeSource(error)]
        scan = self.engine.scan("BTCUSD", force=True)
        self.assertEqual(scan.feed_status.value, "NO_DATA")
        self.assertEqual(scan.consensus, "NO_DATA")
        self.assertEqual(scan.live_sources, 0)
        self.assertEqual(scan.intensity, 0)
        gate = next(item for item in scan.gates if item.id == "G1")
        self.assertEqual(gate.state.value, "FAIL")

    def test_analysis_uses_slm_when_confident(self) -> None:
        analysis = self.engine.analyze_text(
            "Bitcoin breakout with massive volume, buy the dip",
            use_model=True,
            record=False,
        )
        self.assertTrue(analysis.is_relevant)
        self.assertEqual(analysis.stance.value, "BUY")
        self.assertEqual(analysis.engine, "sentiment_slm")
        self.assertAlmostEqual(sum(analysis.probabilities.values()), 1.0, places=2)

    def test_irrelevant_text_is_rejected(self) -> None:
        analysis = self.engine.analyze_text(
            "Our office badminton team practices every Tuesday morning",
            use_model=True,
            record=False,
        )
        self.assertFalse(analysis.is_relevant)
        self.assertEqual(analysis.stance.value, "WAIT")

    def test_engine_status_payload(self) -> None:
        status = self.engine.status()
        self.assertIn("slm", status)
        self.assertTrue(status["slm"]["available"])
        self.assertIn("slm_strictness_modes", status)
        self.assertIn("slm_supported_dims", status)


if __name__ == "__main__":
    unittest.main()

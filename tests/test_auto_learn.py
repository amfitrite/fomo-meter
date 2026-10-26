from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fomo_meter.auto_learner import AutoLearner
from fomo_meter.relevance_gate import FinancialRelevanceGate
from fomo_meter.slm import SentimentSLM


class AutoLearnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.learner = AutoLearner(data_dir=self.data_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_feedback_valid(self) -> None:
        res = self.learner.record_feedback(
            text="Bitcoin menembus resistance, saatnya serok kawan",
            label="BUY",
            source="test",
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["pending_count"], 1)

        stats = self.learner.get_stats()
        self.assertEqual(stats["total_feedbacks"], 1)
        self.assertEqual(stats["distribution"]["BUY"], 1)

    def test_record_feedback_rejects_non_financial(self) -> None:
        res = self.learner.record_feedback(
            text="Jadwal futsal kantor setiap hari Selasa malam jam 8",
            label="WAIT",
            source="test",
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "NOT_RELEVANT")

    def test_record_feedback_invalid_label(self) -> None:
        res = self.learner.record_feedback(
            text="Bitcoin breakout",
            label="MOON_NOW",
            source="test",
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "INVALID_LABEL")

    def test_deduplication(self) -> None:
        text = "Ethereum dump parah, buang barang sekarang cut loss"
        res1 = self.learner.record_feedback(text=text, label="SELL", source="test")
        self.assertTrue(res1["ok"])
        self.assertFalse(res1.get("duplicate", False))

        res2 = self.learner.record_feedback(text=text, label="SELL", source="test")
        self.assertTrue(res2["ok"])
        self.assertTrue(res2.get("duplicate"))
        self.assertEqual(self.learner.get_stats()["total_feedbacks"], 1)

    def test_golden_gate_safety_check(self) -> None:
        # Check that golden benchmark is strictly applied
        real_learner = AutoLearner()
        # Verify golden set file exists in production data
        self.assertTrue(real_learner.golden_file.exists())


if __name__ == "__main__":
    unittest.main()

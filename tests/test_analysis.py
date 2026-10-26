from __future__ import annotations

import time
import unittest

from fomo_meter.schema import Post
from fomo_meter.sources.analysis import aggregate_posts
from fomo_meter.classifier import FastCrowdClassifier
from fomo_meter.relevance_gate import FinancialRelevanceGate


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = FastCrowdClassifier()
        cls.gate = FinancialRelevanceGate()

    def aggregate(self, texts):
        posts = [
            Post(
                id=f"p{index}",
                platform="test",
                author="tester",
                text=text,
                created_at=time.time(),
                url=f"https://example.com/{index}",
                source_name="Test",
            )
            for index, text in enumerate(texts)
        ]
        return aggregate_posts(
            posts,
            classifier=self.classifier,
            relevance_gate=self.gate,
            source_id="test",
            name="Test Source",
            icon="TS",
            kind="Test",
            provenance="https://example.com",
        )

    def test_neutral_posts_do_not_outvote_directional_posts(self) -> None:
        result = self.aggregate([
            "Bitcoin breakout with massive volume, buy the dip",
            "Bitcoin is a digital asset",
            "Bitcoin is a digital asset with a fixed supply",
        ])
        self.assertEqual(result.status.value, "LIVE")
        self.assertEqual(result.stance.value, "BUY")
        self.assertEqual(result.analyzed_count, 3)
        self.assertEqual(result.neutral_count, 2)

    def test_evidence_is_verbatim(self) -> None:
        texts = [
            "Bitcoin breakout with massive volume, buy the dip",
            "Whales are dumping hard, cut loss now",
        ]
        result = self.aggregate(texts)
        self.assertIsNotNone(result.evidence)
        self.assertTrue(any(result.evidence in text for text in texts))
        self.assertIsNotNone(result.evidence_url)

    def test_all_noise_is_no_data(self) -> None:
        result = self.aggregate([
            "Our office badminton team practices every Tuesday",
            "Join our telegram channel for slot gacor maxwin",
        ])
        self.assertEqual(result.status.value, "NO_DATA")
        self.assertEqual(result.analyzed_count, 0)


if __name__ == "__main__":
    unittest.main()

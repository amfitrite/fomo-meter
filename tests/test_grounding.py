from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fomo_meter.grounding import TradingGrounder, GroundingResult
from fomo_meter.schema import Stance


class GroundingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grounder = TradingGrounder()

    def test_extract_candidate_terms(self) -> None:
        text = "Beware of sudden rugpull and liquidation cascade on BTC"
        candidates = self.grounder.extract_candidate_terms(text)
        self.assertIn("rugpull", candidates)
        self.assertIn("liquidation cascade", candidates)

    def test_is_financial_content_true(self) -> None:
        text = "Bitcoin price drops as market liquidity dries up"
        self.assertTrue(self.grounder._is_financial_content(text))

    def test_is_financial_content_false(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        self.assertFalse(self.grounder._is_financial_content(text))

    def test_distill_stance_bearish(self) -> None:
        text = "Whales dumping hard, sudden liquidation cascade and crash"
        stance, conf = self.grounder._distill_stance(text)
        self.assertEqual(stance, Stance.SELL)
        self.assertGreater(conf, 0.7)

    def test_distill_stance_bullish(self) -> None:
        text = "Massive breakout, accumulation surge and rally towards new highs"
        stance, conf = self.grounder._distill_stance(text)
        self.assertEqual(stance, Stance.BUY)
        self.assertGreater(conf, 0.7)

    def test_wikipedia_fallback_mock(self) -> None:
        mock_payload = b'{"title": "Bear Market", "extract": "A bear market is a prolonged period of falling asset prices", "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Bear_market"}}}'
        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_payload
            mock_resp.__enter__.return_value = mock_resp
            mock_url.return_value = mock_resp

            result = self.grounder._search_wikipedia("bear market")
            self.assertIsNotNone(result)
            self.assertEqual(result.term, "bear market")
            self.assertEqual(result.source_engine, "Wikipedia Financial REST API")
            self.assertIn("falling asset prices", result.evidence_quote)


if __name__ == "__main__":
    unittest.main()

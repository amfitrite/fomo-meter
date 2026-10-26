from __future__ import annotations

import unittest

from sentiment_llm.classifier import FastCrowdClassifier
from sentiment_llm.relevance_gate import FinancialRelevanceGate


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = FastCrowdClassifier()
        cls.gate = FinancialRelevanceGate()

    def classify(self, text: str):
        decision = self.gate.evaluate(text)
        self.assertTrue(decision.is_relevant, decision.summary)
        return self.classifier.classify(decision.sanitized_text or text)

    def test_indonesian_accumulation_slang(self) -> None:
        result = self.classify("Serok tipis-tipis kawan, akumulasi mumpung diskon")
        self.assertEqual(result.stance.value, "BUY")
        self.assertGreater(result.intensity, 50)

    def test_indonesian_loss_slang(self) -> None:
        result = self.classify("Waduh boncos parah, nyangkut di pucuk kawan")
        self.assertEqual(result.stance.value, "SELL")

    def test_bull_trap_is_not_bullish(self) -> None:
        result = self.classify("Not bullish at all, this looks like a textbook bull trap")
        self.assertEqual(result.stance.value, "SELL")

    def test_neutral_market(self) -> None:
        result = self.classify("Market is completely sideways and choppy ahead of CPI release, wait and see")
        self.assertEqual(result.stance.value, "WAIT")

    def test_evidence_is_verbatim_substring(self) -> None:
        text = "Bitcoin breakout with massive volume, buy the dip"
        result = self.classify(text)
        self.assertIsNotNone(result.evidence_quote)
        self.assertIn(result.evidence_quote, text)


if __name__ == "__main__":
    unittest.main()

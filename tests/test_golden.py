from __future__ import annotations

import json
import unittest
from pathlib import Path

from fomo_meter.classifier import FastCrowdClassifier
from fomo_meter.relevance_gate import FinancialRelevanceGate

GOLDEN_FILE = Path(__file__).resolve().parent.parent / "data" / "golden_set.json"


class GoldenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = FastCrowdClassifier()
        cls.gate = FinancialRelevanceGate()
        cls.rows = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

    def test_relevance_gate(self) -> None:
        correct = 0
        for row in self.rows:
            decision = self.gate.evaluate(row["text"])
            correct += int(decision.is_relevant == row["expected_relevant"])
        self.assertEqual(correct, len(self.rows), f"relevance {correct}/{len(self.rows)}")

    def test_lexicon_stance(self) -> None:
        checked = 0
        correct = 0
        for row in self.rows:
            expected = row.get("expected_stance")
            if not expected:
                continue
            decision = self.gate.evaluate(row["text"])
            self.assertTrue(decision.is_relevant)
            result = self.classifier.classify(decision.sanitized_text or row["text"])
            checked += 1
            correct += int(result.stance.value == expected)
            self.assertEqual(
                result.stance.value,
                expected,
                f"{row['id']}: expected {expected}, got {result.stance.value}",
            )
        self.assertEqual(checked, 12)
        self.assertEqual(correct, checked)


if __name__ == "__main__":
    unittest.main()

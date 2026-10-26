from __future__ import annotations

import unittest

from fomo_meter.slm import SentimentSLM


class SLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = SentimentSLM()

    def test_model_is_available(self) -> None:
        self.assertTrue(self.model.available, self.model.error)
        self.assertGreater(self.model.samples, 500)

    def test_probabilities_are_normalized(self) -> None:
        prediction = self.model.predict("Bitcoin breakout with massive volume, buy the dip")
        self.assertEqual(prediction.stance.value, "BUY")
        total = sum(prediction.probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=2)
        self.assertGreater(prediction.confidence, 0.5)

    def test_bearish_text(self) -> None:
        prediction = self.model.predict("Whales dumping hard, cut loss now")
        self.assertEqual(prediction.stance.value, "SELL")

    def test_neutral_text(self) -> None:
        prediction = self.model.predict("Sideways ahead of CPI, wait and see")
        self.assertEqual(prediction.stance.value, "WAIT")
        self.assertEqual(prediction.intensity, 50)

    def test_empty_text_is_wait(self) -> None:
        prediction = self.model.predict("")
        self.assertEqual(prediction.stance.value, "WAIT")
        self.assertEqual(prediction.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

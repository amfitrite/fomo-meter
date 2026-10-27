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

    def test_predict_caching(self) -> None:
        text = "Strong rally approaching all time high"
        first = self.model.predict(text)
        second = self.model.predict(text)
        self.assertEqual(first.stance, second.stance)
        self.assertEqual(first.probabilities, second.probabilities)
        self.assertIs(first, second)

    def test_predict_many_matches_individual(self) -> None:
        texts = [
            "Bitcoin breakout with massive volume",
            "Whales dumping hard, cut loss now",
            "Sideways ahead of CPI, wait and see",
        ]
        batch_preds = self.model.predict_many(texts)
        self.assertEqual(len(batch_preds), 3)
        for text, batch_pred in zip(texts, batch_preds):
            individual_pred = self.model.predict(text)
            self.assertEqual(batch_pred.stance, individual_pred.stance)
            self.assertEqual(batch_pred.intensity, individual_pred.intensity)
            self.assertAlmostEqual(batch_pred.confidence, individual_pred.confidence, places=3)

    def test_sarcasm_incongruity_features(self) -> None:
        from fomo_meter.slm_features import feature_weights
        features = feature_weights("Buy the top right here clown lol")
        self.assertIn("sarcasm:mockery_bullish", features)
        self.assertIn("sarcasm:incongruity", features)


if __name__ == "__main__":
    unittest.main()

import unittest

from app.inference import action_from_probability, segment_from_probability


class BusinessRulesTest(unittest.TestCase):
    def test_segment_thresholds(self) -> None:
        self.assertEqual(segment_from_probability(0.71), "high_potential")
        self.assertEqual(segment_from_probability(0.50), "medium_potential")
        self.assertEqual(segment_from_probability(0.20), "needs_nurturing")

    def test_actions_are_business_friendly(self) -> None:
        self.assertIn("senior counselor", action_from_probability(0.90))
        self.assertIn("follow-up", action_from_probability(0.50))
        self.assertIn("foundation support", action_from_probability(0.10))


if __name__ == "__main__":
    unittest.main()

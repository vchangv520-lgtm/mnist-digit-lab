"""评估指标的手算案例：拒绝样本必须留在分母中。"""

import unittest

from evaluate import summarize_predictions, wilson_interval


class EvaluationMetricsTests(unittest.TestCase):
    def test_rejected_input_remains_in_accuracy_and_recall_denominator(self):
        result = summarize_predictions([0, 0, 1, 1, 2], [0, 1, 1, -1, 2])
        self.assertEqual((result["correct"], result["total"], result["rejected_count"]), (3, 5, 1))
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertEqual(result["confusion_matrix"][0][:3], [1, 1, 0])
        self.assertAlmostEqual(result["per_class"][0]["precision"], 1.0)
        self.assertAlmostEqual(result["per_class"][0]["recall"], 0.5)
        self.assertAlmostEqual(result["per_class"][0]["f1"], 2 / 3)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 0.5)
        self.assertAlmostEqual(result["per_class"][1]["recall"], 0.5)
        self.assertEqual(result["per_class"][1]["rejected"], 1)

    def test_wilson_interval_known_case_and_boundaries(self):
        lower, upper = wilson_interval(3, 5)
        self.assertAlmostEqual(lower, 0.2307242813, places=8)
        self.assertAlmostEqual(upper, 0.8823792258, places=8)
        self.assertAlmostEqual(wilson_interval(0, 5)[0], 0.0)
        self.assertAlmostEqual(wilson_interval(5, 5)[1], 1.0)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


if __name__ == "__main__":
    unittest.main()

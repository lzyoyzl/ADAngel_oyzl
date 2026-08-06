import unittest

from adangel.benchmark.metrics import percentile, summarize_samples


class TestMetrics(unittest.TestCase):
    def test_summary(self):
        result = summarize_samples([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["median_ms"], 2.5)
        self.assertEqual(result["iqr_ms"], 1.5)
        self.assertEqual(result["count"], 4)

    def test_percentile(self):
        self.assertEqual(percentile([0.0, 10.0], 25), 2.5)


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("mixed_benchmark", ROOT / "scripts/benchmark_a100_mixed.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestMixedBenchmarkMetrics(unittest.TestCase):
    def test_actual_format_bytes(self):
        # R=1, K128: 64-byte W or 128-byte A, group metadata and output scale.
        self.assertEqual(module.conversion_bytes("o5/x", "weight_conversion", 1, 1, 128), 64 + 1 + 4 + 64 + 4)
        self.assertEqual(module.conversion_bytes("o5/x", "activation_conversion", 1, 1, 128), 128 + 1 + 128 + 4)
        self.assertEqual(module.conversion_bytes("o6/x", "weight_conversion", 1, 1, 128), 64 + 1 + 2 + 4 + 64 + 4)
        self.assertEqual(module.conversion_bytes("o6/x", "activation_conversion", 1, 1, 128), 128 + 1 + 4 + 128 + 4)
        for variant in ("o0", "o3", "o5/x", "o6/x"):
            self.assertEqual(module.conversion_bytes(variant, "total", 16, 32, 256),
                             sum(module.conversion_bytes(variant, s, 16, 32, 256)
                                 for s in ("weight_conversion", "activation_conversion")))

    def test_pair_by_round_not_ratio_of_marginal_medians(self):
        records = []
        for round_id, (o0, candidate) in enumerate(((2., 1.), (10., 10.), (12., 4.))):
            for case, ms in (("o0", o0), ("o5/x", candidate)):
                records.append({"sample_id": "synthetic", "round": round_id, "mode": "compute_only", "case": case,
                                "summary": {"gemm": {"median_ms": ms, "mean_ms": ms, "cv_percent": 1.}},
                                "timing_stable_cv3": True, "mse_vs_o0": 0., "total_timing": "direct"})
        result = module.summarize_records(records)[1]
        self.assertEqual(result["paired_speedup_vs_o0_median"], 2.)
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["median_ms"], 4.)
        with self.assertRaises(ValueError):
            module.summarize_records(records + records[:1])


if __name__ == "__main__":
    unittest.main()

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
    def test_profile_skips_initial_mixed_gemm_but_not_o3(self):
        self.assertEqual(module.profile_spec("o3", 50)["launch_skip"], 50)
        for variant in ("o5", "o6"):
            for tile in module.TILES:
                spec = module.profile_spec(f"{variant}/{tile}", 50)
                self.assertEqual(spec["launch_skip"], 51)
                self.assertEqual(spec["launch_count"], 1)
                self.assertIn("adangel_sm80_split_grouped", spec["kernel_filter"])
        with self.assertRaises(ValueError):
            module.profile_spec("o0", 50)
        with self.assertRaises(ValueError):
            module.profile_spec("o3", -1)

    def test_conversion_total_aligned_without_mutating_native(self):
        payload = {"timings_ms": {"weight_conversion": [1., 100., 101.],
                                 "activation_conversion": [100., 1., 101.],
                                 "total": [99., 99., 199.]}}
        raw, native_total, method = module.aligned_timings(payload, "conversion_only")
        self.assertEqual(raw["total"], [101., 101., 202.])
        self.assertEqual(module.statistics.median(raw["total"]), 101.)
        self.assertNotEqual(module.statistics.median(raw["total"]),
                            sum(module.statistics.median(raw[s]) for s in ("weight_conversion", "activation_conversion")))
        self.assertEqual(native_total, [99., 99., 199.])
        self.assertEqual(payload["timings_ms"]["total"], native_total)
        self.assertEqual(method, "sum_of_batched_stage_samples")
        with self.assertRaises(ValueError):
            module.aligned_timings({"timings_ms": {"total": [1.]}}, "conversion_only")

    def test_direct_total_is_not_reconstructed(self):
        for mode in ("compute_only", "cold", "steady_state"):
            payload = {"timings_ms": {"gemm": [10.], "total": [12.]}}
            if mode != "compute_only":
                payload["timings_ms"]["activation_conversion"] = [1.]
            raw, saved, method = module.aligned_timings(payload, mode)
            self.assertEqual(raw["total"], [12.])
            self.assertEqual(raw, payload["timings_ms"])
            self.assertEqual(saved, [12.])
            self.assertEqual(method, "single_execution_cuda_event")

    def test_native_event_setup_precedes_warmup(self):
        code = (ROOT / "csrc/sm80/mixed_benchmark.cuh").read_text()
        warmup = code.index("for(int i=0;i<warmup;++i)")
        self.assertLess(code.index("std::vector<Pair> weight_marks"), warmup)
        self.assertLess(code.index("std::vector<Mark> marks"), warmup)
        self.assertGreater(code.index("if(weight) measure_conversion"), code.index("cudaEventSynchronize(marks.back()"))
        self.assertNotIn("batch(", code)

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

    def test_conversion_speedup_requires_matching_methods(self):
        records = []
        for case, method, ms in (("o0", "sum_of_batched_stage_samples", 2.),
                                  ("o3", "sum_of_batched_stage_samples", 1.)):
            records.append({"sample_id": "s", "round": 0, "mode": "conversion_only", "case": case,
                            "summary": {"total": {"median_ms": ms, "mean_ms": ms, "cv_percent": 1.}},
                            "timing_stable_cv3": True, "mse_vs_o0": 0., "total_timing": method})
        self.assertEqual(module.summarize_records(records)[1]["paired_speedup_vs_o0_median"], 2.)
        records[0]["total_timing"] = "batched_amortized_cuda_event"
        self.assertIsNone(module.summarize_records(records)[1]["paired_speedup_vs_o0_median"])


if __name__ == "__main__":
    unittest.main()

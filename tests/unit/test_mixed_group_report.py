"""Report contract tests; synthetic metadata, not GPU/performance evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("group_report", ROOT/"scripts/report_a100_o5_o10.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GroupReportTests(unittest.TestCase):
    def fixture(self):
        records, summary = [], []
        for v in ("o5", "o7", "o9", "o6", "o8", "o10"):
            base = "o5" if v in ("o5", "o7", "o9") else "o6"
            for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
                stages = {s: {"median_ms": 1., "mean_ms": 1.1}
                          for s in ("weight_conversion", "activation_conversion", "gemm", "total")}
                records.append(dict(sample_id="s", case=v, mode=mode, round=0,
                    experiment_naming_version=3, timing_contract_version=2,
                    bitwise_equal_validation=True, input_policy="original_fp16_direct_source_quantization",
                    summary=stages, kernel={"implementation": "test_fixture_not_real_kernel"}))
                summary.append(dict(case=v, mode=mode, median_ms=1., mean_ms=1.1,
                    paired_speedup_median=1., cv_failed_records=0, records=1,
                    paired_baseline=base, median_mse_vs_paired_baseline=.01,
                    mean_mse_vs_paired_baseline=.02))
        return {"config.json": dict(samples=1, rounds=1, warmup=50, repeats=200, inner=100),
                "environment.json": dict(commit="fixture", binary_sha256="0"*64),
                "summary.json": dict(correctness_passed=True, raw_fp16_experiment=True, records=summary),
                "results.jsonl": records}

    def render(self, files):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in files.items():
                text = "\n".join(json.dumps(r) for r in value) if name.endswith("jsonl") else json.dumps(value)
                (root/name).write_text(text, encoding="utf-8")
            return module.render(root)

    def test_grouped_four_mode_tables(self):
        text = self.render(self.fixture())
        self.assertIn("GEMM-only / Compute-only", text)
        self.assertIn("Cold 端到端", text)
        self.assertIn("Steady-state 端到端", text)
        self.assertIn("| O9 | O5 |", text)
        self.assertIn("| O10 | O6 |", text)

    def test_incomplete_unverified_or_mislabelled_data_fail(self):
        for change in ("missing_mode", "missing_record", "duplicate", "unverified", "wrong_round", "old_names"):
            files = self.fixture()
            rows = files["results.jsonl"]
            if change == "missing_mode":
                files["results.jsonl"] = [r for r in rows if r["mode"] == "compute_only"]
            elif change == "missing_record":
                rows.pop()
            elif change == "duplicate":
                rows[-1] = copy.deepcopy(rows[0])
            elif change == "unverified":
                rows[0]["bitwise_equal_validation"] = False
            elif change == "wrong_round":
                rows[0]["round"] = 1
            else:
                rows[0]["experiment_naming_version"] = 2
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.render(files)


if __name__ == "__main__":
    unittest.main()

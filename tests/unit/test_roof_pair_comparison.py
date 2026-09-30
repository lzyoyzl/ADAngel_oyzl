import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPARE = runpy.run_path(str(ROOT / "scripts/compare_roof_trace_candidates.py"))["compare"]


class RoofPairComparison(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=sid, variant="o7", mode=mode, round=r, tune=tune,
                     bitwise_equal_production=True, mse_vs_production=0,
                     mse_vs_o0=.1, mse_vs_paired_fp16=.05,
                     summary=dict(median_ms=(r+1)*(2 if tune==14 else 1), cv_percent=8 if r else 1),
                     stage_summaries={"gemm": dict(cv_percent=8 if r else 1)})
                for sid in ("a", "b") for mode in ("compute_only", "cold")
                for r in range(3) for tune in (14, 21)]

    def test_pairing_retains_all_cv_failures(self):
        result = COMPARE(self.rows(), 14, 21, 2, 3, ["o7"], ["compute_only", "cold"])
        for row in result:
            self.assertEqual(row["paired_speedup_median"], 2)
            self.assertEqual(row["paired_speedup_ci95"], [2, 2])
            self.assertEqual(row["reference_median_ms"], 4)
            self.assertEqual(row["candidate_median_ms"], 2)
            self.assertEqual(row["cv"]["21"], dict(records=6, selected_stage_failed=4, any_stage_failed=4))

    def test_reject_incomplete_duplicate_and_changed_output(self):
        rows = self.rows()
        for value in (rows[:-1], rows + [rows[0]]):
            with self.assertRaisesRegex(ValueError, "coverage"):
                COMPARE(value, 14, 21, 2, 3, ["o7"], ["compute_only", "cold"])
        for key, value in (("mse_vs_paired_fp16", .04), ("bitwise_equal_production", False)):
            changed = copy.deepcopy(rows)
            changed[-1][key] = value
            with self.assertRaises(ValueError):
                COMPARE(changed, 14, 21, 2, 3, ["o7"], ["compute_only", "cold"])

    def test_archived_real_trace_reproduces_paired_conclusion(self):
        path = ROOT / "docs/evidence/a100_o378_roof_v14/runs/o378_roof_v14_trace24/results.jsonl"
        result = COMPARE([json.loads(x) for x in path.read_text().splitlines()], 14, 21, 24, 4, ["o7", "o8"], ["compute_only"])
        for row, expected in zip(result, (1.009183724594082, 1.0096551191967615)):
            self.assertAlmostEqual(row["paired_speedup_median"], expected)
            self.assertGreater(row["paired_speedup_ci95"][0], 1)
            self.assertEqual(row["cv"]["21"]["records"], 96)

    def test_cli_reads_actual_completed_schema(self):
        directory = ROOT / "docs/evidence/a100_o378_roof_v14/runs/o378_roof_v14_trace24"
        output = subprocess.check_output([sys.executable, str(ROOT / "scripts/compare_roof_trace_candidates.py"),
            "--input", str(directory), "--reference", "14", "--candidate", "21"], text=True)
        result = json.loads(output)
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["candidate_tune"], 21)
        self.assertEqual(result["source_binary_sha256"], "88b4c5e206eea0135a71e03c22c2c5f7a22ff3f1c17bfc202c9394e0c8c1cf37")

    def test_reassociation_requires_opt_in_and_does_not_relax_old_gate(self):
        rows=self.rows()
        for row in rows:
            if row['tune']==21:
                row.update(tune=24,bitwise_equal_production=False,mse_vs_production=1e-12,
                    max_abs_vs_production=1e-5,mse_vs_semantic_fp64=1e-13,max_abs_vs_semantic_fp64=1e-6,
                    semantic_tolerance_passed=True,mse_regression_passed=True,fp32_reassociated=True,
                    mse_vs_paired_fp16=.050000001)
        args=(14,24,2,3,['o7'],['compute_only','cold'])
        with self.assertRaisesRegex(ValueError,'bitwise'): COMPARE(rows,*args)
        result=COMPARE(rows,*args,allow_reassociation=True)
        self.assertEqual(result[0]['median_mse_vs_paired_fp16'],.050000001)
        self.assertEqual(result[0]['median_reference_mse_vs_paired_fp16'],.05)
        bad=copy.deepcopy(rows);bad[-1]['mse_regression_passed']=False
        with self.assertRaisesRegex(ValueError,'acceptance'): COMPARE(bad,*args,allow_reassociation=True)
        bad=copy.deepcopy(rows);bad[0]['bitwise_equal_production']=False
        with self.assertRaisesRegex(ValueError,'bitwise'): COMPARE(bad,*args,allow_reassociation=True)


if __name__ == "__main__":
    unittest.main()

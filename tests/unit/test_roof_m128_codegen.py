import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("m128_codegen", ROOT / "scripts/probe_roof_m128_codegen.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class M128GateTests(unittest.TestCase):
    def test_only_configuration_assertion_changes(self):
        for name in MOD.HEADERS.values():
            original = (ROOT / "csrc/sm80" / name).read_text()
            candidate = MOD.generated_header(original)
            self.assertEqual(candidate.replace(MOD.NEW, MOD.OLD), original)
            self.assertEqual(candidate.count(MOD.NEW), 1)
        with self.assertRaises(ValueError):
            MOD.generated_header("unexpected source")

    def test_work_model_and_tradeoff(self):
        # Two A planes, one B plane. Two warp partitions in each M/N dimension.
        def supply(m):
            return (2*m*64*2 + 128*64*2) * (4096//m) * (4096//128) * 32
        self.assertEqual(supply(128)*4, supply(64)*3)
        self.assertEqual(128*128//128, 128)  # FP32 acc/thread doubles: 64 -> 128.
        self.assertEqual(3*(2*128*64+128*64)+3*128*4, 75264)
        self.assertEqual(2*(2*128*64+128*64)+2*(128+128)*4, 51200)


class M128ArchivedEvidenceTests(unittest.TestCase):
    """CPU checks of the recorded v57 screen, not new GPU test results."""

    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "docs/evidence/a100_o378_roof_v57"
        cls.run_dir = cls.root / "runs/o378_roof_v57_screen"
        cls.report = cls.root / "reports/o378_roof_v57"
        cls.rows = [json.loads(line) for line in
                    (cls.run_dir / "results.jsonl").read_text().splitlines()]

    def test_raw_event_statistics_and_summary_reproduce(self):
        from benchmark_roof_m128_probe import stats, summary

        self.assertEqual(len(self.rows), 72)
        self.assertEqual({r["sample_id"] for r in self.rows},
                         {f"layer_00_{p}_proj" for p in ("q", "k", "v", "o")})
        for row in self.rows:
            self.assertEqual(len(row["raw_ms"]), 200)
            for key, value in stats(row["raw_ms"]).items():
                self.assertAlmostEqual(value, row["summary"][key], places=12)
        saved = json.loads((self.run_dir / "summary.json").read_text())
        self.assertEqual(summary(self.rows), saved["records"])
        self.assertTrue(saved["no_filtering"])
        self.assertFalse(saved["production_default_changed"])

    def test_aggregation_rejects_missing_and_corrupted_records(self):
        from benchmark_roof_m128_probe import summary

        for rows in (self.rows[:-1], self.rows + [self.rows[0]],
                     [{**self.rows[0], "mse_vs_current_best": 1.0}] + self.rows[1:]):
            with self.assertRaises(ValueError):
                summary(rows)

    def test_resources_and_native_audit_preserved(self):
        audit = json.loads((self.report / "codegen.json").read_text())
        self.assertTrue(audit["control_comparison"]["passed"])
        for variant in audit["variants"].values():
            for entry in variant["entries"].values():
                self.assertTrue(entry["native_u4_s4"])
                self.assertTrue(entry["native_s4_s4"])
                self.assertTrue(entry["all_copies_bypass_l1"])
                self.assertFalse(entry["int8_mma"])
        for row in self.rows:
            candidate = row["m128"] == 1
            resource = row["probe_resources"]
            self.assertEqual(resource["registers_per_thread"], 255 if candidate else 168)
            self.assertEqual(resource["active_blocks_per_sm"], 2 if candidate else 3)
            self.assertEqual(resource["accumulators_per_thread"], 128 if candidate else 64)
            self.assertTrue(row["bitwise_equal_current_best"])
            self.assertEqual(row["mse_vs_current_best"], 0.0)
        for size in (64, 128):
            self.assertTrue((self.report / f"m128_{size}.sass").is_file())

    def test_safety_evidence_is_only_the_recorded_scope(self):
        for stage in ("preflight", "memcheck"):
            result = json.loads((self.report / stage / "validation.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(result["count"], 96)
            self.assertEqual(len(result["checks"]), 96)
            for check in result["checks"]:
                self.assertTrue(check["bitwise_equal_best"])
                self.assertTrue(check["finite_fp32"])
                self.assertLessEqual(check["shape"][0], 256)
                self.assertLessEqual(check["shape"][1], 256)
        self.assertIn("ERROR SUMMARY: 0 errors",
                      (self.report / "memcheck_valid.log").read_text())


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mixed_ncu_summary", ROOT / "scripts/summarize_mixed_ncu.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MixedNcuSummaryTests(unittest.TestCase):
    def test_preserves_units_and_single_kernel_scope(self):
        path = ROOT / "docs/evidence/a100_mixed_ncu_v2/reports/ncu/mixed_g128_v2/o5_raw.csv"
        result = module.read_single_kernel(path)
        self.assertIn("adangel_sm80_split_grouped<128, 256>", result["kernel"])
        self.assertEqual(result["metrics"]["gpu__time_duration.sum"], {"value": 544.704, "unit": "us"})
        self.assertEqual(result["metrics"]["smsp__inst_executed.sum"]["value"], 131596288)

    def test_mixed_formats_use_same_integer_instruction_count(self):
        base = ROOT / "docs/evidence/a100_mixed_ncu_v2/reports/ncu/mixed_g128_v2"
        o5, o6 = (module.read_single_kernel(base / f"{v}_raw.csv") for v in ("o5", "o6"))
        self.assertEqual(o5["metrics"]["smsp__inst_executed.sum"], o6["metrics"]["smsp__inst_executed.sum"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profile_ncu_kernel.py"
SPEC = importlib.util.spec_from_file_location("profile_ncu_kernel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NcuProfileDriverTest(unittest.TestCase):
    def test_production_kernel_filters_are_specific(self) -> None:
        self.assertEqual(set(MODULE.NCU_KERNEL_FILTERS), {"o1", "o2", "o3", "o4"})
        self.assertIn(
            "adangel_o1_register_partial_128x64",
            MODULE.NCU_KERNEL_FILTERS["o1"],
        )
        self.assertIn("cutlass::device_kernel", MODULE.NCU_KERNEL_FILTERS["o2"])
        self.assertIn("adangel_o3_split_tma_ws", MODULE.NCU_KERNEL_FILTERS["o3"])
        self.assertIn("adangel_o4_bitwise_tma_ws", MODULE.NCU_KERNEL_FILTERS["o4"])

    def test_select_sample_requires_one_exact_match(self) -> None:
        manifest = {
            "samples": [
                {"sample_id": "layer_00_q_proj", "file": "layer_00_q_proj.pt"},
                {"sample_id": "layer_00_k_proj", "file": "layer_00_k_proj.pt"},
            ]
        }
        selected = MODULE.select_sample(manifest, "layer_00_q_proj")
        self.assertEqual(selected["file"], "layer_00_q_proj.pt")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.select_sample(manifest, "missing")

    def test_argument_contract(self) -> None:
        args = MODULE.arguments(
            ["--variant", "o4", "--warmup", "7", "--repeats", "1"]
        )
        self.assertEqual(args.variant, "o4")
        self.assertEqual(args.warmup, 7)
        self.assertEqual(args.repeats, 1)
        self.assertEqual(args.sample_id, "layer_00_q_proj")


if __name__ == "__main__":
    unittest.main()

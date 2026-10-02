import importlib.util
import math
from pathlib import Path
import random
import unittest

from adangel.quantization.mixed_formats import BOOKS, decode_scalar, fixed_scalar

PATH = Path(__file__).resolve().parents[2] / "scripts/inspect_o78_integer_alignment_feasibility.py"
SPEC = importlib.util.spec_from_file_location("o78_alignment", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ScaleFeasibilityTests(unittest.TestCase):
    def test_all_scale_codes_exact(self):
        for kind, count in (("e4m3", 127), ("e6m2", 255), ("ue8m0", 255)):
            for code in range(count):
                m, e = MOD.dyadic(code, kind)
                expected = 2.0**(code-127) if kind == "ue8m0" else decode_scalar(code, kind)
                self.assertEqual(math.ldexp(m, e), expected)
                self.assertTrue(m == 0 or m & 1)
        for kind, code in (("ue8m0", 255), ("e6m2", 255), ("e4m3", 127), ("e4m3", 128)):
            with self.assertRaises(ValueError):
                MOD.dyadic(code, kind)

    def test_reachable_partial_bounds(self):
        self.assertEqual(max(fixed_scalar(x, 4, 0) for x in BOOKS["e2m1"]), 6)
        self.assertEqual(max(fixed_scalar(x, 8, -2) for x in BOOKS["e4m3"]), 112)
        self.assertEqual(max(fixed_scalar(x*4, 4, 0) for x in BOOKS["s1p2"]), 7)
        self.assertEqual(max(fixed_scalar(x, 6, 2) for x in BOOKS["e2m3"]), 30)
        self.assertEqual(MOD.PARTIAL_BOUNDS, {"o7": 86016, "o8": 26880})

    def test_arbitrary_precision_and_zero(self):
        self.assertEqual(MOD.exact_bound([(0, 0)], [(7, 0)], 1), 0)
        self.assertEqual(MOD.exact_bound([(1, -127), (1, 127)], [(15, -9)]*2, 86016), 86016*15*(1+2**254))
        a, w = [(1, 1), (1, 0)], [(3, 0), (1, 1)]
        self.assertEqual(MOD.exact_bound(a, w, 1), 4)
        self.assertEqual(MOD.exact_bound(a, w, 1, True), 8)

    def test_vectorized_verdict_matches_python(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed; run on A100 environment")
        rng = random.Random(20261002)
        for device in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
            for variant, ak, wk in (("o7", "ue8m0", "e4m3"), ("o8", "e4m3", "e6m2")):
                for extreme in (False, True):
                    def codes(kind, rows):
                        options = range(127 if kind == "e4m3" else 255) if extreme else range(48, 56)
                        return [[rng.choice(options) for _ in range(32)] for _ in range(rows)]
                    a, w = codes(ak, 5), codes(wk, 7)
                    result = MOD.inspect_codes(torch.tensor(a), torch.tensor(w), variant, device, 2)
                    for anchor, sep in (("tight_output_anchor", False), ("separable_row_column_anchor", True)):
                        bounds = [MOD.exact_bound([MOD.dyadic(c, ak) for c in ar], [MOD.dyadic(c, wk) for c in wr], MOD.PARTIAL_BOUNDS[variant], sep) for ar in a for wr in w]
                        self.assertEqual(result["anchors"][anchor]["int32_safe_outputs"], sum(b <= MOD.INT32_MAX for b in bounds))
                        self.assertEqual(result["anchors"][anchor]["safe_with_bound_at_most_2pow24"], sum(b <= 2**24 for b in bounds))
                        self.assertLessEqual(result["anchors"][anchor]["maximum_capped_bound_lower_bound"], max(bounds))


if __name__ == "__main__":
    unittest.main()

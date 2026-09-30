"""Source gates only; GPU numerical/ISA/performance acceptance is separate."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RoofCandidatesContractTest(unittest.TestCase):
    def test_production_default_unchanged(self):
        host = (ROOT / "csrc/sm80/o1_o3.cu").read_text()
        self.assertIn('implementation="o3_swizzle_64x128_k256_exp_static_stream_bound2_store2";', host)
        self.assertIn('int RoofTune=0>', (ROOT / "csrc/sm80/o3_optimized.cuh").read_text())

    def test_same_group_math_no_magic(self):
        body = (ROOT / "csrc/sm80/o3_optimized.cuh").read_text()
        candidate = body.split("if constexpr(RoofTune!=0) {", 1)[1].split("} else {\n          o1_static_for<0,NAtoms>", 1)[0]
        self.assertIn("const int partial=pl(vi)+16*ph(vi);", candidate)
        self.assertIn("__fmul_rn(row,column)", candidate)
        self.assertIn("__fmaf_rn(float(partial),scale,acc(vi,mi,full_ni))", candidate)
        self.assertNotIn("__int_as_float", candidate)
        self.assertIn("SM80_16x8x64_S32U4S4S32_TN", candidate)
        self.assertIn("SM80_16x8x64_S32S4S4S32_TN", candidate)

    def test_exact_old_entry_and_guarded_shapes(self):
        text = (ROOT / "csrc/sm80/roof_candidates.cuh").read_text()
        self.assertIn("if(tune==-1 && !dual)", text)
        self.assertIn("const bool existing_dual=tune==-1 && dual;", text)
        self.assertIn("adangel_sm80_split_grouped_major<128,256>", text)
        self.assertIn("m64*k64<=2147483647LL", text)
        self.assertIn("as.stride(0)==1", text)
        self.assertIn("ws.ne(255)", text)

    def test_four_mode_opt_in_preserves_defaults(self):
        host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
        mixed=(ROOT/'csrc/sm80/mixed_benchmark.cuh').read_text()
        self.assertEqual(host.count('py::arg("roof_tune")=-1'),2)
        self.assertIn('candidate_not_production',host)
        self.assertIn('candidate_not_production',mixed)
        self.assertIn('no silent fallback',host)
        self.assertIn('roof candidate requires O7/O8 group-major 64x128x256',mixed)


if __name__ == "__main__":
    unittest.main()

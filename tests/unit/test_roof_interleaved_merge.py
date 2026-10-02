import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_roof_interleaved_merge_codegen import generated_header, HEADERS, START, END


class InterleavedMergeTests(unittest.TestCase):
    def test_control_unchanged(self):
        for name in HEADERS.values():
            text = (ROOT / "csrc/sm80" / name).read_text()
            self.assertEqual(generated_header(text, 0), text)

    def test_only_partial_path_changes(self):
        for name in HEADERS.values():
            text = (ROOT / "csrc/sm80" / name).read_text()
            candidate = generated_header(text, 1)
            branch = candidate[candidate.index(START):candidate.index(END)]
            self.assertNotIn("phs", branch)
            self.assertEqual(branch.count("cute::gemm(HA{}"), 2)
            self.assertEqual(branch.count("cute::gemm(LA{}"), 2)
            self.assertIn("pls(vi,ni)*=16", branch)
            self.assertIn("const int partial=pl(vi);", candidate)
            for marker in ("cp.async.commit_group", "__syncthreads", "__fmaf_rn", "__fmul_rn"):
                self.assertEqual(candidate.count(marker), text.count(marker))

    def test_reject_drift(self):
        with self.assertRaises(ValueError):
            generated_header("missing branch", 1)
        with self.assertRaises(ValueError):
            generated_header("anything", 2)

    def test_all_int8_int4_pairs_and_intermediate_bounds(self):
        for a in range(-128, 128):
            lo = a & 15
            hi = a // 16
            for w in range(-8, 8):
                self.assertEqual((hi*w)*16 + lo*w, a*w)
                # Conservative bounds cover every 128-element dot and prefix.
                self.assertLessEqual(abs(hi*w)*128*16 + abs(lo*w)*128, 253952)
        self.assertLess(253952, 2**31)


if __name__ == "__main__": unittest.main()

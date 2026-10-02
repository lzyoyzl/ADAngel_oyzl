import importlib.util
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


if __name__ == "__main__":
    unittest.main()

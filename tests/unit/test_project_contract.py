import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


class TestProjectContract(unittest.TestCase):
    def test_formal_config(self):
        config = yaml.safe_load((ROOT / "configs/experiment/o0_o1_o2_4096.yaml").read_text())
        self.assertEqual([config["matrix"][x] for x in ("m", "n", "k")], [4096, 4096, 4096])
        self.assertEqual(config["output"], {"accumulator": "fp32", "dtype": "fp32"})
        self.assertEqual(config["timing"]["warmup"], 50)
        self.assertEqual(config["timing"]["repeats"], 200)
        self.assertEqual(config["backend"]["cuda_arch"], "sm_120a")

    def test_target_o2_instruction_is_present(self):
        source = (ROOT / "csrc/sm120/o2_microkernel.cu").read_text()
        self.assertIn("kind::mxf4.block_scale.scale_vec::2X", source)
        self.assertIn("m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0", source)


if __name__ == "__main__":
    unittest.main()

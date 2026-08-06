import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TestReferenceVariants(unittest.TestCase):
    def setUp(self):
        import torch

        from adangel.quantization.int8 import quantize_int8_per_row
        from adangel.quantization.mxfp4 import quantize_mxfp4
        from adangel.trace.schema import PreparedInputs

        generator = torch.Generator().manual_seed(7)
        activation = torch.randn((16, 64), generator=generator)
        weight = torch.randn((8, 64), generator=generator)
        a_int8, a_scale = quantize_int8_per_row(activation)
        w_mxfp4, w_scale = quantize_mxfp4(weight)
        self.inputs = PreparedInputs("small", a_int8, a_scale, w_mxfp4, w_scale)

    def test_shapes_finite_and_o0_self_mse(self):
        import torch

        from adangel.benchmark.metrics import mse_fp64
        from adangel.reference import run_o0_reference, run_o1_reference, run_o2_reference

        outputs = [fn(self.inputs)["output"] for fn in (run_o0_reference, run_o1_reference, run_o2_reference)]
        self.assertTrue(all(tuple(output.shape) == (16, 8) for output in outputs))
        self.assertTrue(all(torch.isfinite(output).all() for output in outputs))
        self.assertEqual(mse_fp64(outputs[0], outputs[0]), 0.0)


if __name__ == "__main__":
    unittest.main()

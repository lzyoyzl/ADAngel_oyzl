"""Build-selection regression tests; no CUDA or PyTorch installation required."""
from pathlib import Path
import os
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "db1c288993354c88e551c40c19a8fb93a774a241"


def configuration(target=None, enabled=True):
    torch = ModuleType("torch")
    torch.__version__ = "2.7.1+cu128"
    torch.version = SimpleNamespace(cuda="12.8")
    cpp = ModuleType("torch.utils.cpp_extension")
    cpp.CUDAExtension = lambda name, **kwargs: SimpleNamespace(name=name, **kwargs)
    cpp.BuildExtension = SimpleNamespace(with_options=lambda **kwargs: object())
    setuptools = ModuleType("setuptools")
    setuptools.setup = Mock()
    modules = {
        "torch": torch,
        "torch.utils": ModuleType("torch.utils"),
        "torch.utils.cpp_extension": cpp,
        "setuptools": setuptools,
    }
    env = {"ADANGEL_BUILD_CUDA": "1" if enabled else "0"}
    if target is not None:
        env["ADANGEL_CUDA_TARGET"] = target
    with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, modules), \
         patch.object(Path, "is_file", return_value=True), \
         patch("subprocess.run", return_value=SimpleNamespace(stdout=COMMIT)):
        runpy.run_path(str(ROOT / "setup.py"))
    return setuptools.setup.call_args.kwargs["ext_modules"]


class CudaBuildTargetTests(unittest.TestCase):
    def test_default_retains_all_sm120_sources(self):
        default = configuration()[0]
        explicit = configuration("sm120")[0]
        self.assertEqual(default.name, "adangel._sm120")
        self.assertEqual(default.sources, explicit.sources)
        for filename in ("o0_gemm.cu", "o1_gemm.cu", "o2_cutlass.cu",
                         "o3_gemm.cu", "o3_int8_split_diagnostic.cu", "o4_gemm.cu"):
            self.assertIn("csrc/sm120/" + filename, default.sources)
        self.assertIn("csrc/bindings.cpp", default.sources)
        self.assertFalse(any("csrc/sm80/" in p for p in default.sources))
        self.assertIn("-gencode=arch=compute_120a,code=[sm_120a,compute_120a]",
                      default.extra_compile_args["nvcc"])

    def test_sm80_is_explicit_and_separate(self):
        extension = configuration("sm80")[0]
        self.assertEqual(extension.name, "adangel._sm80")
        self.assertIn("csrc/sm80/o1_o3.cu", extension.sources)
        self.assertNotIn("csrc/bindings.cpp", extension.sources)
        self.assertNotIn("csrc/sm120/o1_gemm.cu", extension.sources)
        self.assertIn("-gencode=arch=compute_80,code=[sm_80,compute_80]",
                      extension.extra_compile_args["nvcc"])

    def test_reject_unknown_target(self):
        with self.assertRaisesRegex(RuntimeError, "ADANGEL_CUDA_TARGET"):
            configuration("unknown")

    def test_trace_only_install_needs_no_extension(self):
        self.assertEqual(configuration(enabled=False), [])


if __name__ == "__main__":
    unittest.main()

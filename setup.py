from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import setup

EXPECTED_CUTLASS_COMMIT = "db1c288993354c88e551c40c19a8fb93a774a241"
EXPECTED_TORCH = "2.7.1"
EXPECTED_TORCH_CUDA = "12.8"


def extensions():
    if os.environ.get("ADANGEL_BUILD_CUDA", "0") != "1":
        return [], {}
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    torch_version = torch.__version__.split("+")[0]
    if torch_version != EXPECTED_TORCH or torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"native build requires torch {EXPECTED_TORCH} with CUDA {EXPECTED_TORCH_CUDA}; "
            f"found torch {torch.__version__} with CUDA {torch.version.cuda}"
        )
    root = Path(__file__).parent
    cutlass_root = Path(os.environ.get("ADANGEL_CUTLASS_ROOT", root / "third_party/cutlass-src"))
    cutlass_header = cutlass_root / "include/cutlass/cutlass.h"
    cutlass_util_header = cutlass_root / "tools/util/include/cutlass/util/packed_stride.hpp"
    if not cutlass_header.is_file():
        raise RuntimeError(
            "Pinned CUTLASS source is missing; run scripts/fetch_cutlass.sh or set ADANGEL_CUTLASS_ROOT"
        )
    if not cutlass_util_header.is_file():
        raise RuntimeError(
            "Pinned CUTLASS utility headers are missing; expected "
            f"{cutlass_util_header}"
        )
    try:
        actual_cutlass = subprocess.run(
            ["git", "-C", str(cutlass_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("CUTLASS must be a git checkout so its pinned revision can be verified") from exc
    if actual_cutlass != EXPECTED_CUTLASS_COMMIT:
        raise RuntimeError(
            f"CUTLASS revision mismatch: expected {EXPECTED_CUTLASS_COMMIT}, got {actual_cutlass}"
        )
    sources = [
        "csrc/bindings.cpp",
        "csrc/common/validation.cu",
        "csrc/sm120/conversion.cu",
        "csrc/sm120/o0_gemm.cu",
        "csrc/sm120/o1_gemm.cu",
        "csrc/sm120/o2_microkernel.cu",
        "csrc/sm120/o2_cutlass.cu",
        "csrc/sm120/o3_gemm.cu",
        "csrc/sm120/o4_gemm.cu",
    ]
    extension = CUDAExtension(
        "adangel._sm120",
        sources=sources,
        include_dirs=[
            str(root / "include"),
            str(cutlass_root / "include"),
            str(cutlass_root / "tools/util/include"),
        ],
        libraries=["cublasLt", "cuda"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-gencode=arch=compute_120a,code=[sm_120a,compute_120a]",
            ],
        },
    )
    return [extension], {"build_ext": BuildExtension.with_options(use_ninja=True)}


ext_modules, cmdclass = extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)

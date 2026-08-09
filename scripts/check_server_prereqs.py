#!/usr/bin/env python3
"""Read-only validation of the pinned RTX 5090 server prerequisites."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

EXPECTED = {
    "python": (3, 10, 12),
    "torch": "2.7.1",
    "torch_cuda": "12.8",
    "nvcc_build": "12.8.93",
    "cuda_home": "/usr/local/cuda-12.8",
    "gcc": "11.5.0",
    "gxx": "11.5.0",
    "git": "2.43.0",
    "cmake": "3.30.5",
    "ninja": "1.11.1",
    "cutlass": "db1c288993354c88e551c40c19a8fb93a774a241",
}
EXPECTED_PACKAGES = {
    "pip": "25.0.1",
    "PyYAML": "6.0.2",
    "typing_extensions": "4.12.2",
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "matplotlib": "3.9.2",
    "pillow": "11.1.0",
    "pytest": "8.3.5",
    "ruff": "0.9.10",
    "setuptools": "75.8.0",
    "wheel": "0.45.1",
    "ninja": "1.11.1.3",
    "cmake": "3.30.5",
}


def command(args: list[str]) -> tuple[bool, str]:
    try:
        process = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (process.stdout + process.stderr).strip()
    return process.returncode == 0, output


def add(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def numeric_version(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    if not parts:
        return ()
    return tuple(int(part) for part in parts[:3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cutlass", action="store_true", help="skip CUTLASS checkout verification")
    parser.add_argument(
        "--cutlass-root",
        default="third_party/cutlass-src",
        help="CUTLASS git checkout (default: third_party/cutlass-src)",
    )
    args = parser.parse_args()
    checks: list[dict] = []

    add(checks, "os", platform.system() == "Linux", platform.platform())
    add(checks, "architecture", platform.machine() == "x86_64", platform.machine())
    os_release = Path("/etc/os-release")
    os_text = os_release.read_text(encoding="utf-8") if os_release.is_file() else ""
    add(
        checks,
        "ubuntu_release",
        'ID=ubuntu' in os_text and 'VERSION_ID="24.04"' in os_text,
        next((line for line in os_text.splitlines() if line.startswith("PRETTY_NAME=")), "unknown"),
    )
    add(
        checks,
        "python",
        sys.version_info[:3] == EXPECTED["python"],
        platform.python_version(),
    )

    cuda_home = os.environ.get("CUDA_HOME", "not set")
    add(
        checks,
        "conda_environment",
        os.environ.get("CONDA_DEFAULT_ENV") == "adangel-sm120",
        os.environ.get("CONDA_DEFAULT_ENV", "not set"),
    )
    add(checks, "CUDA_HOME", cuda_home == EXPECTED["cuda_home"], cuda_home)
    add(checks, "CC", os.environ.get("CC") == "/usr/bin/gcc-11", os.environ.get("CC", "not set"))
    add(checks, "CXX", os.environ.get("CXX") == "/usr/bin/g++-11", os.environ.get("CXX", "not set"))
    add(
        checks,
        "CUDACXX",
        os.environ.get("CUDACXX") == "/usr/local/cuda-12.8/bin/nvcc",
        os.environ.get("CUDACXX", "not set"),
    )
    add(
        checks,
        "TORCH_CUDA_ARCH_LIST",
        os.environ.get("TORCH_CUDA_ARCH_LIST") == "12.0a",
        os.environ.get("TORCH_CUDA_ARCH_LIST", "not set"),
    )
    add(
        checks,
        "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
        os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    )
    library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    add(checks, "cuda_library_path", "/usr/local/cuda-12.8/lib64" in library_paths, str(library_paths))
    for package, expected_version in EXPECTED_PACKAGES.items():
        try:
            actual_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual_version = "not installed"
        add(checks, f"python_package:{package}", actual_version == expected_version, actual_version)

    try:
        import torch
    except ImportError as exc:
        add(checks, "torch", False, str(exc))
        torch = None
    if torch is not None:
        torch_base = torch.__version__.split("+")[0]
        add(checks, "torch", torch_base == EXPECTED["torch"], torch.__version__)
        add(checks, "torch_cuda_build", torch.version.cuda == EXPECTED["torch_cuda"], str(torch.version.cuda))
        cuda_available = torch.cuda.is_available()
        add(checks, "cuda_available", cuda_available, str(cuda_available))
        if cuda_available:
            name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            add(checks, "gpu", "RTX 5090" in name, name)
            add(checks, "compute_capability", capability == (12, 0), str(capability))
            add(checks, "single_visible_gpu", torch.cuda.device_count() == 1, str(torch.cuda.device_count()))

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        add(checks, "nvidia_driver", False, "nvidia-smi not found in PATH")
    else:
        ok, output = command(
            [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"]
        )
        rows = [line.strip() for line in output.splitlines() if line.strip()]
        fields = [value.strip() for value in rows[0].split(",", 1)] if rows else []
        gpu_name = fields[0] if fields else "unparsed"
        driver = fields[1] if len(fields) == 2 else "unparsed"
        driver_ok = numeric_version(driver) >= numeric_version("570.124.06")
        add(checks, "nvidia_smi_single_gpu", ok and len(rows) == 1, str(rows))
        add(checks, "nvidia_smi_gpu", ok and "RTX 5090" in gpu_name, gpu_name)
        add(checks, "nvidia_driver", ok and driver_ok, driver)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        add(checks, "nvcc", False, "not found in PATH")
    else:
        ok, output = command([nvcc, "--version"])
        build = re.search(r"V(\d+\.\d+\.\d+)", output)
        actual_build = build.group(1) if build else "unparsed"
        add(checks, "nvcc", ok and actual_build == EXPECTED["nvcc_build"], actual_build)

    executable_versions = {
        "gcc-11": EXPECTED["gcc"],
        "g++-11": EXPECTED["gxx"],
        "git": EXPECTED["git"],
        "cmake": EXPECTED["cmake"],
        "ninja": EXPECTED["ninja"],
    }
    for executable, expected_version in executable_versions.items():
        path = shutil.which(executable)
        if path is None:
            add(checks, executable, False, "not found in PATH")
            continue
        version_flag = "-dumpfullversion" if executable in {"gcc-11", "g++-11"} else "--version"
        ok, output = command([path, version_flag])
        first = output.splitlines()[0] if output else "no output"
        passed = ok and numeric_version(first) == numeric_version(expected_version)
        add(checks, executable, passed, first)

    if not args.skip_cutlass:
        root = Path(args.cutlass_root)
        core_header = root / "include/cutlass/cutlass.h"
        packed_stride_header = (
            root / "tools/util/include/cutlass/util/packed_stride.hpp"
        )
        add(
            checks, "cutlass_core_headers", core_header.is_file(), str(core_header)
        )
        add(
            checks,
            "cutlass_utility_headers",
            packed_stride_header.is_file(),
            str(packed_stride_header),
        )
        ok, output = command(["git", "-C", str(root), "rev-parse", "HEAD"])
        actual = output.splitlines()[0] if output else "unavailable"
        add(checks, "cutlass_commit", ok and actual == EXPECTED["cutlass"], actual)

    passed = all(item["passed"] for item in checks)
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

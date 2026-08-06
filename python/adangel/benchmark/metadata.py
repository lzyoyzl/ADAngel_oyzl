"""Capture enough environment data to reproduce or reject a run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from ..ops.extension import native_status


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict:
    import torch

    status = native_status()
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    cutlass_root = os.environ.get("ADANGEL_CUTLASS_ROOT", "third_party/cutlass-src")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": _command(["nvcc", "--version"]),
        "cutlass_root": cutlass_root,
        "cutlass_commit": _command(["git", "-C", cutlass_root, "rev-parse", "HEAD"]),
        "gpu": gpu,
        "driver": _command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "nvidia_smi": _command(["nvidia-smi", "-q"]),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        "native": {
            "available": status.available,
            "reason": status.reason,
            "capabilities": status.capabilities,
        },
    }


def write_environment(path: Path) -> None:
    path.write_text(json.dumps(collect_environment(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

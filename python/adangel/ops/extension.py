"""Native extension discovery and strict SM120 capability checks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

EXPECTED_TORCH = "2.7.1"
EXPECTED_TORCH_CUDA = "12.8"


@dataclass(frozen=True)
class NativeStatus:
    available: bool
    reason: str
    capabilities: dict


def load_native():
    try:
        return importlib.import_module("adangel._sm120")
    except (ImportError, OSError):
        return None


def native_status() -> NativeStatus:
    try:
        import torch
    except ImportError:
        return NativeStatus(False, "PyTorch is not installed", {})
    torch_version = torch.__version__.split("+")[0]
    if torch_version != EXPECTED_TORCH:
        return NativeStatus(False, f"requires PyTorch {EXPECTED_TORCH}, found {torch.__version__}", {})
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        return NativeStatus(
            False,
            f"requires PyTorch CUDA {EXPECTED_TORCH_CUDA}, found {torch.version.cuda}",
            {},
        )
    if not torch.cuda.is_available():
        return NativeStatus(False, "CUDA is not available", {})
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "RTX 5090" not in name or capability != (12, 0):
        return NativeStatus(False, f"requires RTX 5090 sm_120, found {name} sm_{capability[0]}{capability[1]}", {})
    extension = load_native()
    if extension is None:
        return NativeStatus(False, "adangel._sm120 is not built", {})
    capabilities = dict(extension.capabilities())
    required = (
        "o0_fp16_tc",
        "o1_int8_tc",
        "o2_mxf4_block_scale",
        "o2_cutlass_tiled",
        "compiled_sm120a",
    )
    missing = [key for key in required if not capabilities.get(key, False)]
    if missing:
        return NativeStatus(False, f"native extension lacks: {', '.join(missing)}", capabilities)
    return NativeStatus(True, "ready", capabilities)


def require_sm120_extension():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "RTX 5090" not in name or capability != (12, 0):
        raise RuntimeError(f"requires RTX 5090 sm_120, found {name} sm_{capability[0]}{capability[1]}")
    extension = load_native()
    if extension is None:
        raise RuntimeError("adangel._sm120 is not built")
    capabilities = dict(extension.capabilities())
    if not capabilities.get("compiled_sm120a", False):
        raise RuntimeError("extension was not compiled for sm_120a")
    return extension


def require_native():
    status = native_status()
    if not status.available:
        raise RuntimeError(f"formal SM120 backend unavailable: {status.reason}")
    return load_native()

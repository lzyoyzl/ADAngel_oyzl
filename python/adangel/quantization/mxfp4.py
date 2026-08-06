"""OCP MXFP4 (E2M1 + UE8M0, K32) reference codec.

Scalar helpers intentionally use only the Python standard library so encoding tests can run on a
machine without PyTorch/CUDA. Tensor helpers import torch lazily.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

E2M1_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_DECODE = E2M1_POSITIVE + tuple(-x for x in E2M1_POSITIVE)
E2M1_TO_INT8_BASE = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
GROUP_SIZE = 32
UE8M0_BIAS = 127
UE8M0_NAN = 255


def decode_e2m1(code: int) -> float:
    if not 0 <= int(code) <= 15:
        raise ValueError(f"E2M1 code must be in [0,15], got {code}")
    return E2M1_DECODE[int(code)]


def encode_e2m1(value: float) -> int:
    """Round to E2M1 with roundTiesToEven and saturating overflow."""
    value = float(value)
    if math.isnan(value):
        raise ValueError("E2M1 has no NaN encoding")
    sign = 8 if math.copysign(1.0, value) < 0.0 and value != 0.0 else 0
    magnitude = abs(value)
    if math.isinf(magnitude) or magnitude >= 6.0:
        return sign | 7
    # The encoding LSB is the significand parity bit for every midpoint. Prefer its even value.
    code = min(range(8), key=lambda i: (abs(E2M1_POSITIVE[i] - magnitude), i & 1))
    return sign | code


def e2m1_to_int8_base(code: int) -> int:
    if not 0 <= int(code) <= 15:
        raise ValueError(f"E2M1 code must be in [0,15], got {code}")
    return E2M1_TO_INT8_BASE[int(code)]


def decode_ue8m0(code: int) -> float:
    code = int(code)
    if not 0 <= code <= 255:
        raise ValueError(f"UE8M0 code must be in [0,255], got {code}")
    if code == UE8M0_NAN:
        return math.nan
    return math.ldexp(1.0, code - UE8M0_BIAS)


def encode_ue8m0_power(exponent: int) -> int:
    """Encode 2**exponent, saturating to finite UE8M0 codes."""
    return max(0, min(254, int(exponent) + UE8M0_BIAS))


def choose_ue8m0_scale_code(max_abs: float) -> int:
    """OCP MX recommended scale for E2M1.

    X is the largest power of two <= max_abs divided by the largest power of two representable
    by E2M1 (4). A zero block uses scale=1 because E8M0 has no zero encoding.
    """
    max_abs = float(max_abs)
    if math.isnan(max_abs) or max_abs < 0.0:
        raise ValueError("max_abs must be finite or +inf and non-negative")
    if max_abs == 0.0:
        return UE8M0_BIAS
    if math.isinf(max_abs):
        return 254
    floor_log2 = math.frexp(max_abs)[1] - 1
    return encode_ue8m0_power(floor_log2 - 2)


def quantize_mxfp4_block(values: Sequence[float]) -> tuple[list[int], int]:
    """Dependency-free reference for exactly one 32-element MXFP4 block."""
    if len(values) != GROUP_SIZE:
        raise ValueError("an MXFP4 block must contain exactly 32 elements")
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("MXFP4 input contains NaN/Inf")
    scale_code = choose_ue8m0_scale_code(max(abs(value) for value in values))
    scale = decode_ue8m0(scale_code)
    return [encode_e2m1(value / scale) for value in values], scale_code


def dequantize_mxfp4_block(codes: Sequence[int], scale_code: int) -> list[float]:
    if len(codes) != GROUP_SIZE:
        raise ValueError("an MXFP4 block must contain exactly 32 elements")
    scale = decode_ue8m0(scale_code)
    if not math.isfinite(scale):
        raise ValueError("UE8M0 NaN scale code is forbidden")
    return [decode_e2m1(code) * scale for code in codes]


def pack_e2m1(codes: Sequence[int]) -> bytes:
    """Pack even K in the low nibble and odd K in the high nibble."""
    if len(codes) % 2:
        raise ValueError("E2M1 packing requires an even number of elements")
    packed = bytearray(len(codes) // 2)
    for i in range(0, len(codes), 2):
        lo, hi = int(codes[i]), int(codes[i + 1])
        if not (0 <= lo <= 15 and 0 <= hi <= 15):
            raise ValueError("E2M1 code must be in [0,15]")
        packed[i // 2] = lo | (hi << 4)
    return bytes(packed)


def unpack_e2m1(packed: Iterable[int]) -> list[int]:
    result: list[int] = []
    for byte in packed:
        byte = int(byte)
        if not 0 <= byte <= 255:
            raise ValueError("packed byte must be in [0,255]")
        result.extend((byte & 0xF, byte >> 4))
    return result


def pack_mma_scale_fragments(sfa: Sequence[Sequence[int]], sfb: Sequence[Sequence[int]]):
    """Pack logical SFA[16,2]/SFB[2,8] for selector {byte-id,thread-id}={0,0}."""
    if len(sfa) != 16 or any(len(row) != 2 for row in sfa):
        raise ValueError("SFA must have logical shape [16,2]")
    if len(sfb) != 2 or any(len(row) != 8 for row in sfb):
        raise ValueError("SFB must have logical shape [2,8]")
    values = [int(x) for row in sfa for x in row] + [int(x) for row in sfb for x in row]
    if any(not 0 <= x <= 254 for x in values):
        raise ValueError("scale fragments require finite UE8M0 codes in [0,254]")
    scale_a = [0] * 32
    scale_b = [0] * 32
    for q in range(8):
        scale_a[4 * q] = int(sfa[q][0]) | (int(sfa[q][1]) << 8)
        scale_a[4 * q + 1] = int(sfa[q + 8][0]) | (int(sfa[q + 8][1]) << 8)
        scale_b[4 * q] = int(sfb[0][q]) | (int(sfb[1][q]) << 8)
    return scale_a, scale_b


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on dependency-light hosts
        raise RuntimeError("Tensor MXFP4 operations require PyTorch") from exc
    return torch


def quantize_mxfp4(x, group_size: int = GROUP_SIZE):
    """Quantize the last dimension and return packed uint8 values plus UE8M0 uint8 scales."""
    torch = _torch()
    if group_size != GROUP_SIZE:
        raise ValueError("SM120 MXFP4 experiment requires group_size=32")
    if x.ndim != 2 or x.shape[1] % group_size:
        raise ValueError("x must be rank-2 and K must be divisible by 32")
    x32 = x.to(torch.float32)
    rows, cols = x32.shape
    blocks = x32.reshape(rows, cols // group_size, group_size)
    amax = blocks.abs().amax(dim=-1)
    if not torch.isfinite(amax).all():
        raise ValueError("MXFP4 input contains NaN/Inf")

    # floor(log2(amax))-2 follows OCP section 6.3; all-zero blocks use exponent 0.
    exponents = torch.floor(torch.log2(torch.where(amax == 0, torch.ones_like(amax), amax))) - 2
    exponents = torch.where(amax == 0, torch.zeros_like(exponents), exponents).clamp(-127, 127)
    scale_codes = (exponents.to(torch.int32) + UE8M0_BIAS).to(torch.uint8)
    scales = torch.pow(2.0, exponents).unsqueeze(-1)
    normalized = blocks / scales

    levels = torch.tensor(E2M1_POSITIVE, dtype=torch.float32, device=x.device)
    distance = (normalized.abs().unsqueeze(-1) - levels).abs()
    minimum = distance.amin(dim=-1, keepdim=True)
    tied = distance == minimum
    even = torch.tensor([not (i & 1) for i in range(8)], dtype=torch.bool, device=x.device)
    tied_even = tied & even
    nearest = torch.argmin(distance, dim=-1)
    even_choice = torch.argmax(tied_even.to(torch.int32), dim=-1)
    codes = torch.where(tied_even.any(dim=-1), even_choice, nearest).to(torch.uint8)
    codes = codes | ((normalized < 0).to(torch.uint8) << 3)
    codes = codes.reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), scale_codes.contiguous()


def unpack_e2m1_tensor(packed):
    torch = _torch()
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed values must be a rank-2 uint8 tensor")
    result = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    result[:, 0::2] = packed & 0xF
    result[:, 1::2] = packed >> 4
    return result


def decode_ue8m0_tensor(scale_codes):
    torch = _torch()
    if scale_codes.dtype != torch.uint8:
        raise ValueError("UE8M0 scales must be uint8")
    if torch.any(scale_codes == UE8M0_NAN):
        raise ValueError("UE8M0 NaN scale code 255 is forbidden")
    return torch.pow(2.0, scale_codes.to(torch.int32) - UE8M0_BIAS)


def decode_e2m1_tensor(codes):
    torch = _torch()
    lut = torch.tensor(E2M1_DECODE, dtype=torch.float32, device=codes.device)
    return lut[codes.to(torch.long)]


def mxfp4_to_int8_base(packed):
    torch = _torch()
    codes = unpack_e2m1_tensor(packed)
    lut = torch.tensor(E2M1_TO_INT8_BASE, dtype=torch.int8, device=packed.device)
    return lut[codes.to(torch.long)].contiguous()


def dequantize_mxfp4(packed, scale_codes, dtype=None):
    torch = _torch()
    values = decode_e2m1_tensor(unpack_e2m1_tensor(packed))
    if values.shape[1] % GROUP_SIZE:
        raise ValueError("unpacked K must be divisible by 32")
    if tuple(scale_codes.shape) != (values.shape[0], values.shape[1] // GROUP_SIZE):
        raise ValueError("scale shape does not match packed values")
    values = values.reshape(values.shape[0], -1, GROUP_SIZE)
    result = (values * decode_ue8m0_tensor(scale_codes).unsqueeze(-1)).reshape(values.shape[0], -1)
    return result.to(dtype or torch.float32)

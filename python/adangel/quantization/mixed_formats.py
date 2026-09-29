"""O7/O8 G128 experimental source formats and fixed-point reference.

FP16 -> source format is common preparation, NOT the timed conversion.
Source format -> fixed integer + compensated scale is the online conversion.
The Torch functions below are correctness references, never a native benchmark.
"""
from __future__ import annotations

import bisect
import math
import struct

GROUP = 128
VERSION = 1
FORMATS = {
    "nvfp4_g128": ("e2m1", 4, 0),
    "mxfp8_e4m3_g128": ("e4m3", 8, -2),
    "hif4_g128": ("s1p2", 4, 0),
    "nvstyle_fp6_e2m3_g128": ("e2m3", 6, 2),
}
VARIANTS = {
    "o7": ("nvfp4_g128", "mxfp8_e4m3_g128"),
    "o8": ("hif4_g128", "nvstyle_fp6_e2m3_g128"),
}
FP16_BASELINES = {"o5": VARIANTS["o7"], "o6": VARIANTS["o8"]}
PAIRED_BASELINE = {"o7": "o5", "o8": "o6"}
BINARY_VARIANTS = {"o9": "o7", "o10": "o8"}
EXPERIMENT_NAMING_VERSION = 3


def positive_codebook(kind: str) -> tuple[float, ...]:
    if kind == "e6m2":
        return tuple(math.ldexp(1 + (c & 3) / 4, (c >> 2) - 48)
                     for c in range(255))  # 255 is NaN; no zero/subnormal.
    if kind == "s1p2":
        return tuple(c / 4 for c in range(8))
    exp_bits, mant_bits, bias = {"e2m1": (2, 1, 1), "e2m3": (2, 3, 1),
                                  "e4m3": (4, 3, 7)}[kind]
    count = (1 << (exp_bits + mant_bits)) - (kind == "e4m3")
    values = []
    for c in range(count):
        exp, mant = c >> mant_bits, c & ((1 << mant_bits) - 1)
        values.append(math.ldexp(mant / (1 << mant_bits), 1 - bias) if exp == 0
                      else math.ldexp(1 + mant / (1 << mant_bits), exp - bias))
    return tuple(values)


BOOKS = {k: positive_codebook(k) for k in ("e2m1", "e2m3", "e4m3", "e6m2", "s1p2")}
SIGN_BITS = {"e2m1": 8, "e2m3": 32, "e4m3": 128, "s1p2": 8}


def decode_scalar(code: int, kind: str) -> float:
    code = int(code)
    sign_bit = SIGN_BITS.get(kind, 0)
    limit = 255 if kind == "e6m2" else sign_bit * 2
    if not 0 <= code < limit:
        raise ValueError(f"invalid {kind} code: {code}")
    index = code & (sign_bit - 1) if sign_bit else code
    if index >= len(BOOKS[kind]):
        raise ValueError(f"{kind} NaN code is forbidden")
    value = BOOKS[kind][index]
    return -value if sign_bit and code & sign_bit else value


def encode_scalar(value: float, kind: str) -> int:
    """Finite, saturating, nearest-even scalar encoder; negative zero canonicalized."""
    value = float(value)
    if not math.isfinite(value) or (kind == "e6m2" and value < 0):
        raise ValueError("encoder requires finite values (E6M2 also nonnegative)")
    levels = BOOKS[kind]
    mids = [(a + b) / 2 for a, b in zip(levels, levels[1:])]
    mag = abs(value)
    i = bisect.bisect_left(mids, mag)
    if i < len(mids) and mag == mids[i] and i & 1:
        i += 1
    return i | (SIGN_BITS.get(kind, 0) if value < 0 else 0)


def fixed_scalar(value: float, q_bits: int, fractional_bits: int) -> int:
    if q_bits not in (4, 6, 8) or not math.isfinite(value):
        raise ValueError("invalid fixed conversion")
    limit = (1 << (q_bits - 1)) - 1
    return max(-limit, min(limit, round(math.ldexp(value, fractional_bits))))


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def hif4_block_scalar(values):
    """Independent G128 reference; preserve upstream BF16 intermediates/half-up leaf.

    Extend upstream G64 to 16 groups of 8 / 32 groups of 4. Return unpacked
    S1P2 codes and micro bits. Final HiF4 -> Q4 uses RNE, not this half-up rule.
    """
    values = [_f32(float(v)) for v in values]
    if len(values) != GROUP or any(not math.isfinite(v) for v in values):
        raise ValueError("requires 128 finite values")
    v4 = [max(abs(v) for v in values[i:i + 4]) for i in range(0, GROUP, 4)]
    v8 = [max(v4[i:i + 2]) for i in range(0, 32, 2)]
    sf = _bf16(_f32(max(v8) * _bf16(1 / 7)))
    scale_code = encode_scalar(min(49152.0, max(2.0 ** -48, sf)), "e6m2")
    scale = decode_scalar(scale_code, "e6m2")
    reciprocal = _bf16(1 / scale)
    micro8 = [int(_f32(v * reciprocal) >= 4) for v in v8]
    micro4 = [int(_f32(_f32(v * reciprocal) * 2.0 ** -micro8[i // 2]) >= 2)
              for i, v in enumerate(v4)]
    codes = []
    for i, v in enumerate(values):
        e = micro8[i // 8] + micro4[i // 4]
        mag = _f32(_f32(abs(v) * reciprocal) * 2.0 ** (2 - e))
        code = min(7, int(math.floor(_f32(mag + 0.5))))
        codes.append(code | (8 if v < 0 else 0))
    return codes, scale_code, micro8, micro4


def _torch():
    import torch
    return torch


def _encode_tensor(x, kind):
    torch = _torch()
    levels = BOOKS[kind]
    mids = torch.tensor([(a + b) / 2 for a, b in zip(levels, levels[1:])],
                        device=x.device, dtype=torch.float32)
    mag = x.abs().contiguous()
    index = torch.bucketize(mag, mids)
    midpoint = mids[index.clamp(max=len(mids) - 1)]
    index = index + ((index < len(mids)) & (mag == midpoint) & ((index & 1) != 0))
    code = index.to(torch.uint8)
    if kind in SIGN_BITS:
        code = code | ((x < 0).to(torch.uint8) * SIGN_BITS[kind])
    return code


def _decode_tensor(codes, kind):
    torch = _torch()
    sign = SIGN_BITS.get(kind, 0)
    index = (codes & (sign - 1)) if sign else codes
    if torch.any(index >= len(BOOKS[kind])):
        raise ValueError(f"{kind} NaN/invalid code")
    lut = torch.tensor(BOOKS[kind], dtype=torch.float32, device=codes.device)
    decoded = lut[index.long()]
    return torch.where((codes & sign) != 0, -decoded, decoded) if sign else decoded


def _pack_nibbles(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _unpack_nibbles(packed):
    torch = _torch()
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def _pack_bits(bits):
    torch = _torch()
    weights = 1 << torch.arange(8, device=bits.device)
    return (bits.reshape(*bits.shape[:-1], -1, 8).long() * weights).sum(-1).to(torch.uint8)


def _unpack_bits(packed):
    torch = _torch()
    shifts = torch.arange(8, device=packed.device)
    return ((packed.long().unsqueeze(-1) >> shifts) & 1).flatten(-2)


def quantize_source(x, fmt: str) -> dict:
    """Common preparation. RNE except upstream HiF4 leaf half-away-from-zero.

    Tensor scale is one scalar per matrix. Zero NV blocks use scale=0 and
    zero payload; MX zero blocks use UE8M0=127; HiF4 zero uses E6M2 minimum.
    MX scale is upward-rounded amax/448, as in the documented TE recipe.
    """
    torch = _torch()
    if fmt not in FORMATS:
        raise ValueError(f"unknown experimental source format: {fmt}")
    if x.ndim != 2 or not x.is_floating_point() or min(x.shape) <= 0 or x.shape[1] % GROUP:
        raise ValueError("source must be floating rank two with positive G128-aligned K")
    x = x.to(torch.float32).contiguous()
    if not torch.isfinite(x).all():
        raise ValueError("source contains NaN/Inf")
    rows, k = x.shape
    blocks = x.reshape(rows, k // GROUP, GROUP)
    amax = blocks.abs().amax(-1)
    result = {"version": VERSION, "format": fmt, "group_size": GROUP, "shape": [rows, k]}
    kind, _, _ = FORMATS[fmt]
    if fmt == "hif4_g128":
        # Upstream G64 reference uses BF16(1/7), BF16(scale), BF16(reciprocal).
        inv7 = torch.tensor(1 / 7, device=x.device).bfloat16().float()
        sf = (amax * inv7).bfloat16().float().clamp(2.0 ** -48, 49152)
        result["scale"] = _encode_tensor(sf, "e6m2").contiguous()
        sf = _decode_tensor(result["scale"], "e6m2")
        rec = sf.reciprocal().bfloat16().float()
        v4 = blocks.abs().reshape(rows, -1, 32, 4).amax(-1)
        v8 = v4.reshape(rows, -1, 16, 2).amax(-1)
        e8 = (v8 * rec[..., None] >= 4).to(torch.uint8)
        e4 = (v4 * rec[..., None] * torch.exp2(-e8.repeat_interleave(2, -1).float()) >= 2).to(torch.uint8)
        e = e8.repeat_interleave(8, -1) + e4.repeat_interleave(4, -1)
        magnitude = blocks.abs() * rec[..., None] * torch.exp2(2 - e.float())
        codes = torch.floor(magnitude + 0.5).clamp(0, 7).to(torch.uint8)
        codes = codes | ((blocks < 0).to(torch.uint8) << 3)
        result["micro8"] = _pack_bits(e8).contiguous()  # two bytes per G128
        result["micro4"] = _pack_bits(e4).contiguous()  # four bytes per G128
    elif fmt == "mxfp8_e4m3_g128":
        ratio = torch.where(amax == 0, torch.ones_like(amax), amax / 448)
        exp = torch.ceil(torch.log2(ratio)).clamp(-127, 127)
        # Verify rounding direction near powers of two, independent of log2 accuracy.
        exp = torch.where(torch.exp2(exp) < ratio, exp + 1, exp).clamp(-127, 127)
        result["scale"] = (exp.to(torch.int32) + 127).to(torch.uint8).contiguous()
        codes = _encode_tensor(blocks / torch.exp2(exp)[..., None], kind)
    else:
        maximum = BOOKS[kind][-1]
        tensor_max = amax.max()
        global_scale = torch.where(tensor_max == 0, torch.ones_like(tensor_max),
                                   tensor_max / (448 * maximum)).reshape(1)
        if not torch.isfinite(global_scale).all() or not (global_scale > 0).all():
            raise ValueError("tensor scale outside positive FP32 range")
        result["tensor_scale"] = global_scale.contiguous()
        block_scale = (amax / maximum) / global_scale
        result["scale"] = _encode_tensor(block_scale, "e4m3").contiguous()
        effective = _decode_tensor(result["scale"], "e4m3") * global_scale
        local = torch.where(effective[..., None] == 0, torch.zeros_like(blocks),
                            blocks / torch.where(effective == 0, torch.ones_like(effective), effective)[..., None])
        codes = _encode_tensor(local, kind)
    codes = codes.reshape(rows, k)
    result["payload"] = (_pack_nibbles(codes) if kind in ("e2m1", "s1p2") else codes.contiguous())
    validate_source(result)
    return result


def validate_source(source: dict) -> None:
    torch = _torch()
    fmt = source.get("format")
    if fmt not in FORMATS or source.get("version") != VERSION or source.get("group_size") != GROUP:
        raise ValueError("unknown source format/version/group size")
    shape = source.get("shape")
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("invalid source shape")
    rows, k = shape
    if k % GROUP:
        raise ValueError("source K must be G128 aligned")
    kind = FORMATS[fmt][0]
    expected = {"payload": ((rows, k // 2 if kind in ("e2m1", "s1p2") else k), torch.uint8),
                "scale": ((rows, k // GROUP), torch.uint8)}
    if fmt == "hif4_g128":
        expected.update(micro8=((rows, k // GROUP, 2), torch.uint8),
                        micro4=((rows, k // GROUP, 4), torch.uint8))
    elif fmt.startswith(("nvfp4", "nvstyle")):
        expected["tensor_scale"] = ((1,), torch.float32)
    device = source["payload"].device if isinstance(source.get("payload"), torch.Tensor) else None
    for name, (dims, dtype) in expected.items():
        value = source.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != dims or value.dtype != dtype:
            raise ValueError(f"invalid {name} shape/dtype")
        if not value.is_contiguous() or value.device != device:
            raise ValueError("source tensors must be contiguous on the same device")
    if fmt in ("hif4_g128", "mxfp8_e4m3_g128"):
        if torch.any(source["scale"] == 255):
            raise ValueError("NaN scale code 255")
    else:
        if torch.any(source["scale"] > 126):
            raise ValueError("NV scale must be nonnegative finite E4M3")
        gs = source["tensor_scale"]
        if not torch.isfinite(gs).all() or not (gs > 0).all():
            raise ValueError("invalid tensor scale")
    if kind == "e4m3" and torch.any((source["payload"] & 127) == 127):
        raise ValueError("NaN E4M3 payload")
    if kind == "e2m3" and torch.any(source["payload"] > 63):
        raise ValueError("invalid E2M3 storage bits")


def source_components(source: dict):
    """Return local values [R,G,128] and effective source scales [R,G]."""
    torch = _torch()
    validate_source(source)
    fmt = source["format"]
    kind = FORMATS[fmt][0]
    rows, k = source["shape"]
    codes = source["payload"]
    if kind in ("e2m1", "s1p2"):
        codes = _unpack_nibbles(codes)
    local = _decode_tensor(codes, kind).reshape(rows, k // GROUP, GROUP)
    if fmt == "hif4_g128":
        exp = (_unpack_bits(source["micro8"]).repeat_interleave(8, -1)
               + _unpack_bits(source["micro4"]).repeat_interleave(4, -1))
        local = local * torch.exp2(exp.float())
        scale = _decode_tensor(source["scale"], "e6m2")
    elif fmt == "mxfp8_e4m3_g128":
        scale = torch.exp2(source["scale"].float() - 127)
    else:
        scale = _decode_tensor(source["scale"], "e4m3") * source["tensor_scale"]
    if not torch.isfinite(scale).all():
        raise ValueError("effective source scale overflow")
    return local, scale.contiguous()


def dequantize_source(source: dict):
    local, scale = source_components(source)
    return (local * scale[..., None]).reshape(source["shape"]).contiguous()


def to_fixed_reference(source: dict):
    torch = _torch()
    local, scale = source_components(source)
    _, bits, fractional = FORMATS[source["format"]]
    limit = (1 << (bits - 1)) - 1
    integers = torch.round(local * 2.0 ** fractional).clamp(-limit, limit).to(torch.int8)
    # int8 is already sign-extended for Q6; never zero-pad raw six-bit codes.
    integers = integers.reshape(source["shape"]).contiguous()
    effective = (scale * 2.0 ** -fractional).contiguous()
    if not torch.isfinite(effective).all():
        raise ValueError("fixed scale overflow")
    return integers, effective


def prepare_integer_reference(weight: dict, activation: dict):
    from .arbitrary_bits import split_int8_to_packed_int4
    if (weight.get("format"), activation.get("format")) not in VARIANTS.values():
        raise ValueError("requires the approved O7 or O8 source-format pair")
    wq, ws = to_fixed_reference(weight)
    aq, asc = to_fixed_reference(activation)
    if wq.shape[1] != aq.shape[1] or wq.device != aq.device:
        raise ValueError("weight and activation must have the same K and device")
    return {"A_split": split_int8_to_packed_int4(aq), "A_scale": asc,
            "W_q4": _pack_nibbles(wq.to(_torch().uint8) & 15), "W_scale": ws}

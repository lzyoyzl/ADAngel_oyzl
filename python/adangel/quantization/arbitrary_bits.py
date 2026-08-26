"""Q4 Split and two's-complement bit-plane reference utilities."""

from __future__ import annotations


A8_BIT_WEIGHTS = (1, 2, 4, 8, 16, 32, 64, -128)
Q4_BIT_WEIGHTS = (1, 2, 4, -8)


def split_int8_scalar(value: int) -> tuple[int, int]:
    """Return the exact (UINT4 low, INT4 high) Split representation."""
    value = int(value)
    if not -128 <= value <= 127:
        raise ValueError("INT8 value must be in [-128,127]")
    raw = value & 0xFF
    low = raw & 0xF
    high_raw = raw >> 4
    high = high_raw - 16 if high_raw & 0x8 else high_raw
    return low, high


def int8_bitplanes_scalar(value: int) -> tuple[int, ...]:
    value = int(value)
    if not -128 <= value <= 127:
        raise ValueError("INT8 value must be in [-128,127]")
    raw = value & 0xFF
    return tuple((raw >> plane) & 1 for plane in range(8))


def q4_bitplanes_scalar(value: int) -> tuple[int, ...]:
    value = int(value)
    if not -8 <= value <= 7:
        raise ValueError("Q4 value must be in [-8,7]")
    raw = value & 0xF
    return tuple((raw >> plane) & 1 for plane in range(4))


def reconstruct_from_bitplanes(bits, weights) -> int:
    if len(bits) != len(weights):
        raise ValueError("bit and weight counts must match")
    if any(int(bit) not in (0, 1) for bit in bits):
        raise ValueError("bit-plane values must be zero or one")
    return sum(int(bit) * int(weight) for bit, weight in zip(bits, weights))


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tensor arbitrary-bit operations require PyTorch") from exc
    return torch


def split_int8_to_packed_int4(values):
    """Pack A8 as the paper Split layout [A_low_u4; A_high_s4].

    The returned uint8 tensor has shape [2*M, K/2]. The first M rows are
    unsigned low nibbles and the second M rows are signed-high nibbles stored
    as raw two's-complement Q4 codes.
    """
    torch = _torch()
    if values.ndim != 2 or values.dtype != torch.int8 or values.shape[1] % 2:
        raise ValueError("values must be rank-2 INT8 with even K")
    raw = values.to(torch.int16) & 0xFF
    low = raw & 0xF
    high = (raw >> 4) & 0xF

    def pack(nibbles):
        return (
            nibbles[:, 0::2].to(torch.uint8)
            | (nibbles[:, 1::2].to(torch.uint8) << 4)
        ).contiguous()

    return torch.cat((pack(low), pack(high)), dim=0).contiguous()


def unpack_split_int4(packed, rows: int):
    """Return UINT4 low and signed INT4 high tensors from a stacked layout."""
    torch = _torch()
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed Split activation must be rank-2 uint8")
    if packed.shape[0] != 2 * rows:
        raise ValueError("packed Split activation must contain 2*M rows")

    def unpack(part):
        result = torch.empty(
            (part.shape[0], part.shape[1] * 2),
            dtype=torch.uint8,
            device=part.device,
        )
        result[:, 0::2] = part & 0xF
        result[:, 1::2] = part >> 4
        return result

    low = unpack(packed[:rows])
    high_raw = unpack(packed[rows:]).to(torch.int16)
    high = torch.where(high_raw >= 8, high_raw - 16, high_raw).to(torch.int8)
    return low.contiguous(), high.contiguous()


def pack_twos_complement_bitplanes(values, bit_width: int):
    """Pack each bit position along K into signed int32 storage words.

    Shape: values[R,K] -> words[bit_width,R,K/32]. The int32 words are raw
    b32 containers; negative host values simply denote that bit 31 is set.
    """
    torch = _torch()
    if values.ndim != 2 or values.dtype != torch.int8:
        raise ValueError("values must be a rank-2 int8 tensor")
    if bit_width not in (4, 8):
        raise ValueError("bit_width must be 4 or 8")
    if values.shape[1] % 32:
        raise ValueError("K must be divisible by 32 for bit-plane packing")
    raw = values.to(torch.int64) & ((1 << bit_width) - 1)
    planes = torch.arange(bit_width, dtype=torch.int64, device=values.device)
    bits = ((raw.unsqueeze(0) >> planes[:, None, None]) & 1).reshape(
        bit_width, values.shape[0], values.shape[1] // 32, 32
    )
    word_weights = (1 << torch.arange(32, dtype=torch.int64, device=values.device))
    return (bits * word_weights).sum(dim=-1).to(torch.int32).contiguous()


def unpack_twos_complement_bitplanes(words, bit_width: int):
    """Unpack natural b32 planes and reconstruct signed INT8/Q4 values."""
    torch = _torch()
    if words.ndim != 3 or words.dtype != torch.int32:
        raise ValueError("words must be rank-3 int32 [planes,rows,K/32]")
    if words.shape[0] != bit_width or bit_width not in (4, 8):
        raise ValueError("bit-plane count must match bit_width 4 or 8")
    raw_words = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(32, dtype=torch.int64, device=words.device)
    bits = ((raw_words.unsqueeze(-1) >> shifts) & 1).reshape(
        bit_width, words.shape[1], words.shape[2] * 32
    )
    weights = A8_BIT_WEIGHTS if bit_width == 8 else Q4_BIT_WEIGHTS
    coefficients = torch.tensor(weights, dtype=torch.int64, device=words.device)
    return (bits * coefficients[:, None, None]).sum(dim=0).to(torch.int8).contiguous()

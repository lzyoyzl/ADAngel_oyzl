"""O4 paper-Bitwise reference using two's-complement B1 planes."""

from __future__ import annotations

from ..quantization.arbitrary_bits import (
    A8_BIT_WEIGHTS,
    Q4_BIT_WEIGHTS,
    pack_twos_complement_bitplanes,
)
from ..quantization.mxfp4 import ARBITRARY_BIT_GROUP_SIZE, decode_ue8m0_tensor
from .o3 import _q4_weight


def _logical_planes(values, bit_width: int):
    import torch

    raw = values.to(torch.int16) & ((1 << bit_width) - 1)
    shifts = torch.arange(bit_width, dtype=torch.int16, device=values.device)
    return ((raw.unsqueeze(0) >> shifts[:, None, None]) & 1).to(torch.int32)


def run_o4_reference(inputs):
    import torch

    if getattr(inputs, "W_scale_g128", None) is None:
        raise ValueError("O4 inputs require W_scale_g128")
    packed_weight, weight = _q4_weight(inputs)
    activation_words = pack_twos_complement_bitplanes(inputs.A_int8, 8)
    weight_words = pack_twos_complement_bitplanes(weight, 4)
    activation_planes = _logical_planes(inputs.A_int8, 8)
    weight_planes = _logical_planes(weight, 4)
    m, k = inputs.A_int8.shape
    n = weight.shape[0]
    if k % ARBITRARY_BIT_GROUP_SIZE:
        raise ValueError("O4 requires K divisible by 128")
    output = torch.zeros((m, n), dtype=torch.float32, device=inputs.A_int8.device)
    weight_scales = decode_ue8m0_tensor(inputs.W_scale_g128)
    for group in range(k // ARBITRARY_BIT_GROUP_SIZE):
        sl = slice(
            group * ARBITRARY_BIT_GROUP_SIZE,
            (group + 1) * ARBITRARY_BIT_GROUP_SIZE,
        )
        group_accumulator = torch.zeros(
            (m, n), dtype=torch.int32, device=inputs.A_int8.device
        )
        for a_plane, a_weight in enumerate(A8_BIT_WEIGHTS):
            a_bits = activation_planes[a_plane, :, sl]
            for w_plane, w_weight in enumerate(Q4_BIT_WEIGHTS):
                w_bits = weight_planes[w_plane, :, sl].transpose(0, 1)
                partial = a_bits @ w_bits
                group_accumulator.add_(partial * (a_weight * w_weight))
        scale = inputs.A_scale[:, None] * weight_scales[:, group][None, :]
        output.add_(group_accumulator.to(torch.float32) * scale)
    return {
        "output": output,
        "converted_weight": (packed_weight, weight_words),
        "converted_activation": activation_words,
        "weight_conversion_ms": 0.0,
        "activation_conversion_ms": 0.0,
        "gemm_ms": 0.0,
        "total_ms": 0.0,
    }

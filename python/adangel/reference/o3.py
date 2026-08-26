"""O3 paper-Split reference adapted to G128 block scales."""

from __future__ import annotations

from ..quantization.arbitrary_bits import split_int8_to_packed_int4, unpack_split_int4
from ..quantization.mxfp4 import (
    ARBITRARY_BIT_GROUP_SIZE,
    decode_ue8m0_tensor,
    mxfp4_to_q4_packed,
    unpack_int4_tensor,
)


def _q4_weight(inputs):
    packed = getattr(inputs, "W_q4", None)
    if packed is None:
        source = getattr(inputs, "W_mxfp4_g128", None)
        if source is None:
            raise ValueError("O3/O4 inputs require W_mxfp4_g128 or offline W_q4")
        packed = mxfp4_to_q4_packed(source)
    return packed, unpack_int4_tensor(packed)


def run_o3_reference(inputs):
    import torch

    if getattr(inputs, "W_scale_g128", None) is None:
        raise ValueError("O3 inputs require W_scale_g128")
    packed_activation = split_int8_to_packed_int4(inputs.A_int8)
    low, high = unpack_split_int4(packed_activation, inputs.A_int8.shape[0])
    packed_weight, weight = _q4_weight(inputs)
    m, k = inputs.A_int8.shape
    n = weight.shape[0]
    if k % ARBITRARY_BIT_GROUP_SIZE:
        raise ValueError("O3 requires K divisible by 128")
    output = torch.zeros((m, n), dtype=torch.float32, device=inputs.A_int8.device)
    weight_scales = decode_ue8m0_tensor(inputs.W_scale_g128)
    for group in range(k // ARBITRARY_BIT_GROUP_SIZE):
        sl = slice(
            group * ARBITRARY_BIT_GROUP_SIZE,
            (group + 1) * ARBITRARY_BIT_GROUP_SIZE,
        )
        w_group = weight[:, sl].to(torch.int32).transpose(0, 1)
        low_partial = low[:, sl].to(torch.int32) @ w_group
        high_partial = high[:, sl].to(torch.int32) @ w_group
        partial = low_partial + 16 * high_partial
        scale = inputs.A_scale[:, None] * weight_scales[:, group][None, :]
        output.add_(partial.to(torch.float32) * scale)
    return {
        "output": output,
        "converted_weight": packed_weight,
        "converted_activation": packed_activation,
        "weight_conversion_ms": 0.0,
        "activation_conversion_ms": 0.0,
        "gemm_ms": 0.0,
        "total_ms": 0.0,
    }

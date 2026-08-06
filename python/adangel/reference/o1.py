"""O1 semantic reference with one INT32 dot and FP32 rescale per K32 group."""

from __future__ import annotations

from ..quantization.mxfp4 import GROUP_SIZE, decode_ue8m0_tensor, mxfp4_to_int8_base


def run_o1_reference(inputs):
    import torch

    w_int8 = mxfp4_to_int8_base(inputs.W_mxfp4)
    m, k = inputs.A_int8.shape
    n = w_int8.shape[0]
    output = torch.zeros((m, n), dtype=torch.float32, device=inputs.A_int8.device)
    w_scales = decode_ue8m0_tensor(inputs.W_scale)
    for group in range(k // GROUP_SIZE):
        sl = slice(group * GROUP_SIZE, (group + 1) * GROUP_SIZE)
        partial = inputs.A_int8[:, sl].to(torch.int32) @ w_int8[:, sl].to(torch.int32).transpose(0, 1)
        scale = inputs.A_scale[:, None] * w_scales[:, group][None, :] * 0.5
        output.add_(partial.to(torch.float32) * scale)
    return {
        "output": output,
        "converted_weight": w_int8,
        "converted_activation": None,
        "weight_conversion_ms": 0.0,
        "activation_conversion_ms": 0.0,
        "gemm_ms": 0.0,
        "total_ms": 0.0,
    }

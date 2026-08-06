"""O2 semantic reference: requantize A to MXFP4, then block-scaled FP32 dot products."""

from __future__ import annotations

from ..quantization.int8 import dequantize_int8_per_row
from ..quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4


def run_o2_reference(inputs):
    import torch

    activation = dequantize_int8_per_row(inputs.A_int8, inputs.A_scale, torch.float32)
    a_mxfp4, a_scale = quantize_mxfp4(activation)
    a_dequant = dequantize_mxfp4(a_mxfp4, a_scale, torch.float32)
    w_dequant = dequantize_mxfp4(inputs.W_mxfp4, inputs.W_scale, torch.float32)
    output = a_dequant @ w_dequant.transpose(0, 1)
    return {
        "output": output,
        "converted_weight": None,
        "converted_activation": (a_mxfp4, a_scale),
        "weight_conversion_ms": 0.0,
        "activation_conversion_ms": 0.0,
        "gemm_ms": 0.0,
        "total_ms": 0.0,
    }

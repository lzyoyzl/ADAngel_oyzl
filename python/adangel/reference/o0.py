"""O0 semantic reference: dequantize both operands to FP16, accumulate/output FP32."""

from __future__ import annotations

from ..quantization.int8 import dequantize_int8_per_row
from ..quantization.mxfp4 import dequantize_mxfp4


def run_o0_reference(inputs):
    import torch

    a_fp16 = dequantize_int8_per_row(inputs.A_int8, inputs.A_scale, torch.float16)
    w_fp16 = dequantize_mxfp4(inputs.W_mxfp4, inputs.W_scale, torch.float16)
    # Casting the already-rounded FP16 operands to FP32 models FP32 Tensor Core accumulation.
    output = a_fp16.float() @ w_fp16.float().transpose(0, 1)
    return {
        "output": output,
        "converted_weight": w_fp16,
        "converted_activation": a_fp16,
        "weight_conversion_ms": 0.0,
        "activation_conversion_ms": 0.0,
        "gemm_ms": 0.0,
        "total_ms": 0.0,
    }

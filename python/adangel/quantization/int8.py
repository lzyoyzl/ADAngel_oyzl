"""Symmetric per-row INT8 reference quantization."""

from __future__ import annotations


def quantize_int8_per_row(x):
    import torch

    if x.ndim != 2:
        raise ValueError("activation must be rank-2")
    x32 = x.to(torch.float32)
    if not torch.isfinite(x32).all():
        raise ValueError("activation contains NaN/Inf")
    amax = x32.abs().amax(dim=1)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
    quantized = torch.round(x32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scale.to(torch.float32).contiguous()


def dequantize_int8_per_row(quantized, scale, dtype=None):
    import torch

    if quantized.dtype != torch.int8 or quantized.ndim != 2:
        raise ValueError("quantized activation must be rank-2 int8")
    if scale.dtype != torch.float32 or tuple(scale.shape) != (quantized.shape[0],):
        raise ValueError("activation scale must be row-shaped fp32")
    result = quantized.to(torch.float32) * scale[:, None]
    return result.to(dtype or torch.float32)

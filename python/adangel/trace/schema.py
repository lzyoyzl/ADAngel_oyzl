"""Prepared sample schema and fail-fast validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedInputs:
    sample_id: str
    A_int8: object
    A_scale: object
    W_mxfp4: object
    W_scale: object

    @property
    def shape(self) -> tuple[int, int, int]:
        m, k = self.A_int8.shape
        n = self.W_mxfp4.shape[0]
        return int(m), int(n), int(k)


def validate_prepared(inputs: PreparedInputs, formal: bool = False) -> None:
    import torch

    tensors = (inputs.A_int8, inputs.A_scale, inputs.W_mxfp4, inputs.W_scale)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("prepared fields must be torch.Tensor instances")
    m, n, k = inputs.shape
    if formal and (m, n, k) != (4096, 4096, 4096):
        raise ValueError(f"formal experiment requires 4096^3, got {(m, n, k)}")
    if k % 64:
        raise ValueError("K must be divisible by 64 for the O2 MMA")
    expected = {
        "A_int8": ((m, k), torch.int8),
        "A_scale": ((m,), torch.float32),
        "W_mxfp4": ((n, k // 2), torch.uint8),
        "W_scale": ((n, k // 32), torch.uint8),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(inputs, name)
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name}: expected {shape}/{dtype}, got {tuple(tensor.shape)}/{tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if torch.any(inputs.W_scale == 255):
        raise ValueError("UE8M0 NaN scale code 255 is forbidden")

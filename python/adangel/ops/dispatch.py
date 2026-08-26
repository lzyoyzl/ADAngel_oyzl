"""Public experiment interfaces and backend dispatch."""

from __future__ import annotations

from ..reference import (
    run_o0_reference,
    run_o1_reference,
    run_o2_reference,
    run_o3_reference,
    run_o4_reference,
)
from ..trace.prepare import prepare_trace
from ..trace.schema import validate_prepared
from .extension import require_variant

_REFERENCES = {
    "o0": run_o0_reference,
    "o1": run_o1_reference,
    "o2": run_o2_reference,
    "o3": run_o3_reference,
    "o4": run_o4_reference,
}


def _run(inputs, variant: str, mode: str = "cold", backend: str = "native"):
    validate_prepared(inputs, require_arbitrary_bits=variant in {"o3", "o4"})
    if mode not in {"conversion_only", "compute_only", "cold", "steady_state"}:
        raise ValueError(f"unknown timing mode: {mode}")
    if backend == "reference":
        if inputs.shape == (4096, 4096, 4096):
            raise RuntimeError("reference backend is forbidden for formal 4096^3 timing")
        return _REFERENCES[variant](inputs)
    if backend != "native":
        raise ValueError(f"unknown backend: {backend}")
    extension = require_variant(variant)
    weight = inputs.W_mxfp4_g128 if variant in {"o3", "o4"} else inputs.W_mxfp4
    scale = inputs.W_scale_g128 if variant in {"o3", "o4"} else inputs.W_scale
    return getattr(extension, f"run_{variant}")(
        inputs.A_int8, inputs.A_scale, weight, scale, mode
    )


def run_o0(inputs, mode="cold", backend="native"):
    return _run(inputs, "o0", mode, backend)


def run_o1(inputs, mode="cold", backend="native"):
    return _run(inputs, "o1", mode, backend)


def run_o2(inputs, mode="cold", backend="native"):
    return _run(inputs, "o2", mode, backend)


def run_o3(inputs, mode="cold", backend="native"):
    return _run(inputs, "o3", mode, backend)


def run_o4(inputs, mode="cold", backend="native"):
    return _run(inputs, "o4", mode, backend)


def benchmark_variant(
    inputs,
    variant,
    mode,
    warmup=50,
    repeats=200,
    backend="native",
    conversion_inner_repeats=100,
):
    validate_prepared(
        inputs,
        formal=backend == "native",
        require_arbitrary_bits=variant in {"o3", "o4"},
    )
    if backend != "native":
        raise RuntimeError("performance benchmark records require the native SM120 backend")
    extension = require_variant(variant)
    weight = inputs.W_mxfp4_g128 if variant in {"o3", "o4"} else inputs.W_mxfp4
    scale = inputs.W_scale_g128 if variant in {"o3", "o4"} else inputs.W_scale
    return extension.benchmark(
        variant,
        mode,
        inputs.A_int8,
        inputs.A_scale,
        weight,
        scale,
        int(warmup),
        int(repeats),
        int(conversion_inner_repeats),
    )

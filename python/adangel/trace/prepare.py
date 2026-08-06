"""Convert a validated external FP16 trace to the common experiment inputs."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..quantization.int8 import quantize_int8_per_row
from ..quantization.mxfp4 import quantize_mxfp4
from .schema import PreparedInputs
from .storage import save_prepared, write_manifest


def _config(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    return yaml.safe_load(Path(value).read_text(encoding="utf-8"))


def _validate_configs(experiment: dict, trace: dict) -> tuple[list[int], list[str]]:
    matrix = experiment.get("matrix", {})
    if [matrix.get("m"), matrix.get("n"), matrix.get("k")] != [4096, 4096, 4096]:
        raise ValueError("trace preparation requires M=N=K=4096")
    quantization = experiment.get("quantization", {})
    if quantization.get("activation", {}).get("format") != "int8_symmetric_per_row":
        raise ValueError("activation quantization must be int8_symmetric_per_row")
    weight = quantization.get("weight", {})
    if weight.get("format") != "mxfp4_e2m1_ue8m0" or weight.get("group_size") != 32:
        raise ValueError("weight quantization must be MXFP4 E2M1/UE8M0 K32")
    layers = [int(value) for value in trace.get("layers", [])]
    projections = list(trace.get("projections", []))
    if layers != [0, 6, 12, 18, 24, 31] or projections != ["q_proj", "k_proj", "v_proj", "o_proj"]:
        raise ValueError("trace config must select the planned six layers and four projections")
    if trace.get("batch_size") != 1 or trace.get("sequence_length") != 4096:
        raise ValueError("trace config requires batch=1 and 4096 valid tokens")
    if list(trace.get("expected_shape", [])) != [4096, 4096]:
        raise ValueError("trace config expected_shape must be [4096,4096]")
    return layers, projections


def _load_and_validate_raw(path: Path, layer: int, projection: str):
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"sample_id", "layer", "projection", "activation_fp16", "weight_fp16"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        raise ValueError(f"{path.name}: missing required raw trace fields")
    sample_id = f"layer_{layer:02d}_{projection}"
    if raw["sample_id"] != sample_id or raw["layer"] != layer or raw["projection"] != projection:
        raise ValueError(f"{path.name}: sample metadata does not match its configured identity")
    for name in ("activation_fp16", "weight_fp16"):
        tensor = raw[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path.name}: {name} must be a torch.Tensor")
        if tuple(tensor.shape) != (4096, 4096) or tensor.dtype != torch.float16:
            raise ValueError(f"{path.name}: {name} must be [4096,4096] torch.float16")
        if not tensor.is_contiguous():
            raise ValueError(f"{path.name}: {name} must be contiguous")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path.name}: {name} contains NaN/Inf")
    return raw


def prepare_trace(
    input_dir: str | Path,
    output_dir: str | Path,
    experiment_config: dict | str | Path,
    trace_config: dict | str | Path,
) -> list[dict]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    experiment, trace = _config(experiment_config), _config(trace_config)
    layers, projections = _validate_configs(experiment, trace)
    expected = {
        f"layer_{layer:02d}_{projection}.pt": (layer, projection)
        for layer in layers
        for projection in projections
    }
    actual = {path.name for path in input_dir.glob("*.pt")}
    missing, extra = sorted(set(expected) - actual), sorted(actual - set(expected))
    if missing or extra:
        raise ValueError(f"raw trace file set mismatch; missing={missing}, extra={extra}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty prepared directory: {output_dir}")

    # Validate all external files before creating any prepared output.
    for filename, (layer, projection) in expected.items():
        _load_and_validate_raw(input_dir / filename, layer, projection)

    records: list[dict] = []
    for filename, (layer, projection) in expected.items():
        raw = _load_and_validate_raw(input_dir / filename, layer, projection)
        a_int8, a_scale = quantize_int8_per_row(raw["activation_fp16"])
        w_mxfp4, w_scale = quantize_mxfp4(raw["weight_fp16"])
        record = save_prepared(
            PreparedInputs(raw["sample_id"], a_int8, a_scale, w_mxfp4, w_scale), output_dir
        )
        record.update({"layer": layer, "projection": projection})
        records.append(record)

    write_manifest(
        output_dir,
        records,
        {
            "matrix_shape": [4096, 4096, 4096],
            "dtypes": {
                "A_int8": "torch.int8",
                "A_scale": "torch.float32",
                "W_mxfp4": "torch.uint8",
                "W_scale": "torch.uint8",
            },
            "quantization": {
                "activation": "int8_symmetric_per_row",
                "weight": "mxfp4_e2m1_ue8m0_k32",
                "rounding": "round_ties_to_even",
            },
            "trace": {
                "source": "external_fp16_prefill_trace",
                "batch_size": 1,
                "valid_tokens": 4096,
                "layers": layers,
                "projections": projections,
            },
        },
    )
    return records

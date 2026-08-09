"""Convert a validated external FP16 trace to the common experiment inputs."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..quantization.int8 import quantize_int8_per_row
from ..quantization.mxfp4 import quantize_mxfp4
from .raw import RAW_MANIFEST_NAME, validate_raw_trace, validate_trace_config
from .schema import PreparedInputs
from .storage import save_prepared, sha256_file, write_manifest


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
    layers, projections, _, _ = validate_trace_config(trace)
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


def _source_trace_provenance(input_dir: Path, raw_manifest: dict) -> dict:
    tokenization = raw_manifest["tokenization"]
    return {
        "manifest_file": RAW_MANIFEST_NAME,
        "manifest_sha256": sha256_file(input_dir / RAW_MANIFEST_NAME),
        "format": raw_manifest["format"],
        "dataset": raw_manifest["dataset"],
        "tokenization": {
            "seed": tokenization["seed"],
            "start_index": tokenization["start_index"],
            "sampled_tokens": tokenization["sampled_tokens"],
            "prepend_bos": tokenization["prepend_bos"],
            "bos_token_id": tokenization["bos_token_id"],
            "corpus_token_count": tokenization["corpus_token_count"],
            "input_ids_sha256": tokenization["input_ids_sha256"],
        },
        "model": raw_manifest["model"],
        "runtime": raw_manifest["runtime"],
        "inference": raw_manifest["inference"],
        "environment_source": raw_manifest.get("environment_source", {}),
    }


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty prepared directory: {output_dir}")

    # Validate manifest, transfer hashes, tensors, and q/k/v hook identity before
    # creating any prepared output.
    raw_manifest = validate_raw_trace(input_dir, trace, deep=True)

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
            "source_trace": _source_trace_provenance(input_dir, raw_manifest),
        },
    )
    return records

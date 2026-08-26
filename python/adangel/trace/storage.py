"""Trace persistence, manifest validation, and content hashing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .schema import PreparedInputs, validate_prepared

MANIFEST_VERSION = 3
PREPARED_FORMAT = "adangel-prepared-mxfp4-k32-g128-q4"
LEGACY_MANIFEST_VERSION = 2
LEGACY_PREPARED_FORMAT = "adangel-prepared-mxfp4-k32"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_prepared(inputs: PreparedInputs, output_dir: Path) -> dict:
    import torch

    validate_prepared(inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{inputs.sample_id}.pt"
    payload = {
        "sample_id": inputs.sample_id,
        "A_int8": inputs.A_int8.cpu(),
        "A_scale": inputs.A_scale.cpu(),
        "W_mxfp4": inputs.W_mxfp4.cpu(),
        "W_scale": inputs.W_scale.cpu(),
        "shape": list(inputs.shape),
    }
    for name in ("W_mxfp4_g128", "W_scale_g128", "W_q4"):
        tensor = getattr(inputs, name)
        if tensor is not None:
            payload[name] = tensor.cpu()
    torch.save(payload, path)
    return {
        "sample_id": inputs.sample_id,
        "file": path.name,
        "sha256": sha256_file(path),
        "shape": list(inputs.shape),
    }


def load_prepared(path: Path, device: str = "cuda") -> PreparedInputs:
    import torch

    record = torch.load(path, map_location=device, weights_only=True)
    inputs = PreparedInputs(
        record["sample_id"],
        record["A_int8"],
        record["A_scale"],
        record["W_mxfp4"],
        record["W_scale"],
        record.get("W_mxfp4_g128"),
        record.get("W_scale_g128"),
        record.get("W_q4"),
    )
    validate_prepared(inputs)
    if list(inputs.shape) != list(record.get("shape", [])):
        raise ValueError(f"{path}: embedded shape does not match tensors")
    return inputs


def validate_manifest(
    manifest: dict,
    *,
    formal: bool = False,
    require_arbitrary_bits: bool = False,
) -> None:
    identity = (manifest.get("version"), manifest.get("format"))
    legacy = identity == (LEGACY_MANIFEST_VERSION, LEGACY_PREPARED_FORMAT)
    extended = identity == (MANIFEST_VERSION, PREPARED_FORMAT)
    if not (legacy or extended):
        raise ValueError("unsupported prepared manifest version/format")
    if require_arbitrary_bits and not extended:
        raise ValueError("O3/O4 require the extended G128/Q4 prepared manifest")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest samples must be a non-empty list")
    ids, files = set(), set()
    for sample in samples:
        required = {"sample_id", "file", "sha256", "shape", "layer", "projection"}
        if not isinstance(sample, dict) or not required.issubset(sample):
            raise ValueError("manifest sample is missing required fields")
        sample_id, filename = sample["sample_id"], sample["file"]
        if sample_id in ids or filename in files:
            raise ValueError("manifest contains duplicate sample_id or filename")
        if filename != f"{sample_id}.pt" or Path(filename).name != filename:
            raise ValueError(f"unsafe or mismatched sample filename: {filename}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(sample["sha256"])):
            raise ValueError(f"invalid SHA-256 for {sample_id}")
        ids.add(sample_id)
        files.add(filename)

    if not formal:
        return
    expected_shape = [4096, 4096, 4096]
    if manifest.get("matrix_shape") != expected_shape:
        raise ValueError(f"formal manifest must use shape {expected_shape}")
    if any(sample["shape"] != expected_shape for sample in samples):
        raise ValueError("formal sample shape mismatch")
    expected_dtypes = {
        "A_int8": "torch.int8",
        "A_scale": "torch.float32",
        "W_mxfp4": "torch.uint8",
        "W_scale": "torch.uint8",
    }
    if extended:
        expected_dtypes.update(
            {
                "W_mxfp4_g128": "torch.uint8",
                "W_scale_g128": "torch.uint8",
                "W_q4": "torch.uint8",
            }
        )
    if manifest.get("dtypes") != expected_dtypes:
        raise ValueError("formal manifest dtype contract mismatch")
    quantization = manifest.get("quantization", {})
    if quantization.get("activation") != "int8_symmetric_per_row" or quantization.get("weight") != "mxfp4_e2m1_ue8m0_k32":
        raise ValueError("formal manifest quantization contract mismatch")
    if extended and (
        quantization.get("arbitrary_bit_weight")
        != "mxfp4_e2m1_ue8m0_k128_to_q4_rne"
    ):
        raise ValueError("formal manifest G128/Q4 quantization contract mismatch")
    trace = manifest.get("trace", {})
    layers = trace.get("layers")
    projections = trace.get("projections")
    if layers != [0, 6, 12, 18, 24, 31] or projections != ["q_proj", "k_proj", "v_proj", "o_proj"]:
        raise ValueError("formal trace selection mismatch")
    expected_ids = {f"layer_{layer:02d}_{projection}" for layer in layers for projection in projections}
    if len(samples) != 24 or ids != expected_ids:
        raise ValueError("formal manifest must contain exactly the 24 configured samples")


def write_manifest(output_dir: Path, samples: list[dict], metadata: dict) -> None:
    manifest = {
        "version": MANIFEST_VERSION,
        "format": PREPARED_FORMAT,
        **metadata,
        "samples": samples,
    }
    validate_manifest(manifest, formal=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

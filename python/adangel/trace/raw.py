"""Raw FP16 trace manifest helpers and cross-server validation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

RAW_TRACE_VERSION = 1
RAW_TRACE_FORMAT = "adangel-raw-fp16-prefill-trace"
RAW_MANIFEST_NAME = "trace_manifest.json"
FORMAL_LAYERS = [0, 6, 12, 18, 24, 31]
FORMAL_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _config(value: dict | str | Path) -> dict:
    if isinstance(value, dict):
        return value
    import yaml

    return yaml.safe_load(Path(value).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    canonical = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_trace_config(config: dict) -> tuple[list[int], list[str], int, tuple[int, int]]:
    layers = [int(value) for value in config.get("layers", [])]
    projections = list(config.get("projections", []))
    sequence_length = int(config.get("sequence_length", 0))
    expected_shape = tuple(int(value) for value in config.get("expected_shape", []))
    if layers != FORMAL_LAYERS:
        raise ValueError(f"trace layers must be {FORMAL_LAYERS}")
    if projections != FORMAL_PROJECTIONS:
        raise ValueError(f"trace projections must be {FORMAL_PROJECTIONS}")
    if config.get("batch_size") != 1 or sequence_length != 4096:
        raise ValueError("formal trace requires batch=1 and sequence_length=4096")
    if config.get("valid_tokens") != 4096:
        raise ValueError("formal trace requires exactly 4096 valid tokens")
    if str(config.get("dtype", "")).lower() not in {"fp16", "float16"}:
        raise ValueError("formal trace dtype must be fp16")
    if expected_shape != (4096, 4096):
        raise ValueError("formal trace expected_shape must be [4096,4096]")

    calibration = config.get("calibration", {})
    required = {
        "dataset": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "seed": 20250805,
        "sampled_tokens": 4095,
        "prepend_bos": True,
    }
    for key, expected in required.items():
        if calibration.get(key) != expected:
            raise ValueError(
                f"calibration.{key} must be {expected!r}, got {calibration.get(key)!r}"
            )
    if calibration.get("corpus_joiner") != "\n\n":
        raise ValueError(r"calibration.corpus_joiner must be \n\n")
    return layers, projections, sequence_length, expected_shape


def expected_sample_ids(config: dict) -> list[str]:
    layers, projections, _, _ = validate_trace_config(config)
    return [
        f"layer_{layer:02d}_{projection}"
        for layer in layers
        for projection in projections
    ]


def _validate_manifest_contract(manifest: dict, config: dict) -> dict[str, dict]:
    layers, projections, sequence_length, expected_shape = validate_trace_config(config)
    if manifest.get("version") != RAW_TRACE_VERSION:
        raise ValueError("unsupported raw trace manifest version")
    if manifest.get("format") != RAW_TRACE_FORMAT:
        raise ValueError("unsupported raw trace manifest format")

    trace = manifest.get("trace", {})
    expected_trace = {
        "batch_size": 1,
        "sequence_length": sequence_length,
        "valid_tokens": sequence_length,
        "dtype": "fp16",
        "layers": layers,
        "projections": projections,
        "expected_shape": list(expected_shape),
    }
    if trace != expected_trace:
        raise ValueError("raw trace manifest does not match the formal trace configuration")

    calibration = config["calibration"]
    dataset = manifest.get("dataset", {})
    for key in ("dataset", "subset", "split", "revision"):
        if dataset.get(key) != calibration[key]:
            raise ValueError(f"raw trace dataset {key} does not match configuration")
    if not isinstance(dataset.get("fingerprint"), str) or not dataset["fingerprint"]:
        raise ValueError("raw trace dataset fingerprint is missing")
    if dataset.get("corpus_joiner") != calibration["corpus_joiner"]:
        raise ValueError("raw trace corpus joiner mismatch")
    if not isinstance(dataset.get("row_count"), int) or dataset["row_count"] <= 0:
        raise ValueError("raw trace dataset row count is invalid")
    if not _is_sha256(dataset.get("corpus_sha256")):
        raise ValueError("raw trace corpus SHA-256 is invalid")

    tokenization = manifest.get("tokenization", {})
    token_ids = tokenization.get("input_ids")
    if not isinstance(token_ids, list) or len(token_ids) != sequence_length:
        raise ValueError("raw trace must record exactly 4096 input token IDs")
    if any(not isinstance(value, int) or value < 0 for value in token_ids):
        raise ValueError("raw trace input token IDs must be non-negative integers")
    if tokenization.get("input_ids_sha256") != token_ids_sha256(token_ids):
        raise ValueError("raw trace input token SHA-256 mismatch")
    if tokenization.get("seed") != calibration["seed"]:
        raise ValueError("raw trace token seed mismatch")
    if tokenization.get("sampled_tokens") != calibration["sampled_tokens"]:
        raise ValueError("raw trace sampled-token count mismatch")
    if tokenization.get("prepend_bos") is not True:
        raise ValueError("raw trace must prepend BOS")
    bos_token_id = tokenization.get("bos_token_id")
    if not isinstance(bos_token_id, int) or token_ids[0] != bos_token_id:
        raise ValueError("raw trace first token is not the recorded BOS")
    if token_ids.count(bos_token_id) != 1:
        raise ValueError("raw trace must contain exactly one BOS token")
    if tokenization.get("attention_mask_all_ones") is not True:
        raise ValueError("raw trace attention mask must contain only valid tokens")
    start_index = tokenization.get("start_index")
    corpus_tokens = tokenization.get("corpus_token_count")
    if (
        not isinstance(start_index, int)
        or not isinstance(corpus_tokens, int)
        or start_index < 0
        or start_index + calibration["sampled_tokens"] > corpus_tokens
    ):
        raise ValueError("raw trace token window is outside the corpus")

    inference = manifest.get("inference", {})
    expected_inference = {
        "dtype": "fp16",
        "attention_implementation": config.get("attention_implementation", "sdpa"),
        "use_cache": False,
        "local_files_only": True,
        "base_model_only": True,
    }
    if inference != expected_inference:
        raise ValueError("raw trace inference settings do not match the formal contract")

    environment_source = manifest.get("environment_source", {})
    if environment_source != config.get("environment_source", {}):
        raise ValueError("raw trace environment source does not match configuration")

    runtime = manifest.get("runtime", {})
    required_runtime = {
        "python",
        "platform",
        "torch",
        "torch_cuda",
        "cuda_available",
        "gpu",
        "gpu_total_memory_bytes",
        "compute_capability",
        "driver",
        "transformers",
        "datasets",
        "accelerate",
        "pyyaml",
        "safetensors",
        "sentencepiece",
    }
    if not required_runtime.issubset(runtime):
        raise ValueError("raw trace runtime metadata is incomplete")
    if runtime.get("cuda_available") is not True:
        raise ValueError("raw trace was not collected with CUDA")
    if not str(runtime.get("python", "")).startswith("3.10."):
        raise ValueError("formal trace collection requires Python 3.10")
    if runtime.get("torch_cuda") != "12.8":
        raise ValueError("formal trace collection requires a cu128 PyTorch build")
    if "H100" not in str(runtime.get("gpu", "")):
        raise ValueError("formal trace collection requires the H100 model server")
    if int(runtime.get("gpu_total_memory_bytes", 0)) < 70 * 1024**3:
        raise ValueError("formal trace collection requires the H100 80GB server")
    package_names = (
        "transformers",
        "datasets",
        "accelerate",
        "pyyaml",
        "safetensors",
        "sentencepiece",
    )
    if any(str(runtime.get(name, "")).lower() == "missing" for name in package_names):
        raise ValueError("raw trace runtime is missing a required collection package")
    if str(runtime["transformers"]).split(".", 1)[0] != "4":
        raise ValueError("formal trace collection requires Transformers major version 4")

    model = manifest.get("model", {})
    if model.get("model_type") != "llama":
        raise ValueError("raw trace model_type must be llama")
    for key, value in {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }.items():
        if model.get(key) != value:
            raise ValueError(f"raw trace model {key} must be {value}")
    if int(model.get("max_position_embeddings", 0)) < sequence_length:
        raise ValueError("raw trace model context length is below 4096")
    artifact_hashes = model.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("raw trace model artifact hashes are missing")
    if any(not _is_sha256(value) for value in artifact_hashes.values()):
        raise ValueError("raw trace model artifact SHA-256 is invalid")
    if "config.json" not in artifact_hashes:
        raise ValueError("raw trace model config hash is missing")
    if not any(
        name.endswith((".safetensors", ".bin")) for name in artifact_hashes
    ):
        raise ValueError("raw trace model weight-shard hashes are missing")
    if not any(
        name in artifact_hashes
        for name in ("tokenizer.model", "tokenizer.json")
    ):
        raise ValueError("raw trace tokenizer artifact hash is missing")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != len(layers) * len(projections):
        raise ValueError("raw trace manifest must contain exactly 24 samples")
    entries: dict[str, dict] = {}
    expected_ids = set(expected_sample_ids(config))
    for entry in samples:
        required = {
            "sample_id",
            "file",
            "layer",
            "projection",
            "sha256",
            "activation",
            "weight",
        }
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError("raw trace sample entry is incomplete")
        sample_id = entry["sample_id"]
        if sample_id in entries:
            raise ValueError(f"duplicate raw trace sample {sample_id}")
        if sample_id not in expected_ids:
            raise ValueError(f"unexpected raw trace sample {sample_id}")
        if entry["file"] != f"{sample_id}.pt":
            raise ValueError(f"raw trace filename mismatch for {sample_id}")
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"unsafe raw trace filename for {sample_id}")
        expected_id = f"layer_{int(entry['layer']):02d}_{entry['projection']}"
        if expected_id != sample_id:
            raise ValueError(f"raw trace identity mismatch for {sample_id}")
        for tensor_name in ("activation", "weight"):
            metadata = entry[tensor_name]
            if metadata != {"shape": list(expected_shape), "dtype": "torch.float16"}:
                raise ValueError(f"raw trace {sample_id} {tensor_name} metadata mismatch")
        if not _is_sha256(entry["sha256"]):
            raise ValueError(f"raw trace {sample_id} SHA-256 is invalid")
        entries[sample_id] = entry
    if set(entries) != expected_ids:
        raise ValueError("raw trace manifest sample set mismatch")
    return entries


def validate_raw_trace(
    input_dir: str | Path,
    config: dict | str | Path,
    *,
    deep: bool = True,
) -> dict:
    input_dir = Path(input_dir)
    config = _config(config)
    manifest_path = input_dir / RAW_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing raw trace manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _validate_manifest_contract(manifest, config)

    expected_files = {entry["file"] for entry in entries.values()}
    actual_files = {path.name for path in input_dir.glob("*.pt")}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"raw trace file set mismatch; missing={missing}, extra={extra}")

    for sample_id, entry in entries.items():
        path = input_dir / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"raw trace file SHA-256 mismatch: {sample_id}")

    if not deep:
        return manifest

    import torch

    expected_shape = tuple(config["expected_shape"])
    layers = [int(value) for value in config["layers"]]
    projections = list(config["projections"])
    for layer in layers:
        q_activation = None
        for projection in projections:
            sample_id = f"layer_{layer:02d}_{projection}"
            path = input_dir / entries[sample_id]["file"]
            record = torch.load(path, map_location="cpu", weights_only=True)
            required = {
                "sample_id",
                "layer",
                "projection",
                "activation_fp16",
                "weight_fp16",
            }
            if not isinstance(record, dict) or not required.issubset(record):
                raise ValueError(f"{sample_id}: raw payload is incomplete")
            if (
                record["sample_id"] != sample_id
                or record["layer"] != layer
                or record["projection"] != projection
            ):
                raise ValueError(f"{sample_id}: raw payload identity mismatch")
            for name in ("activation_fp16", "weight_fp16"):
                tensor = record[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{sample_id}: {name} must be a tensor")
                if tuple(tensor.shape) != expected_shape or tensor.dtype != torch.float16:
                    raise ValueError(
                        f"{sample_id}: {name} must be {expected_shape}/torch.float16"
                    )
                if not tensor.is_contiguous():
                    raise ValueError(f"{sample_id}: {name} must be contiguous")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{sample_id}: {name} contains NaN/Inf")
            if projection == "q_proj":
                q_activation = record["activation_fp16"]
            elif projection in {"k_proj", "v_proj"}:
                if q_activation is None or not torch.equal(
                    q_activation, record["activation_fp16"]
                ):
                    raise ValueError(
                        f"layer {layer}: q/k/v projection inputs are not identical"
                    )
        del q_activation
    return manifest

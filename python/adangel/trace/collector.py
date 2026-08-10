"""Collect the formal Llama-2-7B WikiText FP16 prefill trace."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .raw import (
    RAW_MANIFEST_NAME,
    RAW_TRACE_FORMAT,
    RAW_TRACE_VERSION,
    sha256_file,
    token_ids_sha256,
    validate_raw_trace,
    validate_trace_config,
)


def build_corpus(texts: list[str], joiner: str) -> str:
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("WikiText rows must be a non-empty list of strings")
    corpus = joiner.join(texts)
    if not corpus:
        raise ValueError("WikiText corpus is empty")
    return corpus


def select_prefill_tokens(
    corpus_token_ids: list[int],
    *,
    bos_token_id: int,
    seed: int,
    sampled_tokens: int,
) -> tuple[list[int], int]:
    if bos_token_id < 0:
        raise ValueError("tokenizer bos_token_id is invalid")
    if len(corpus_token_ids) < sampled_tokens:
        raise ValueError(
            f"WikiText produced {len(corpus_token_ids)} tokens, "
            f"but {sampled_tokens} are required"
        )
    start = random.Random(seed).randrange(len(corpus_token_ids) - sampled_tokens + 1)
    window = [int(value) for value in corpus_token_ids[start : start + sampled_tokens]]
    if bos_token_id in window:
        raise ValueError(
            "add_special_tokens=False unexpectedly produced a BOS token inside the corpus"
        )
    return [bos_token_id, *window], start


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return values[0] if values else None


def _runtime_metadata(torch: Any, device: Any) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "driver": _driver_version(),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "accelerate": _package_version("accelerate"),
        "pyyaml": _package_version("PyYAML"),
        "safetensors": _package_version("safetensors"),
        "sentencepiece": _package_version("sentencepiece"),
        "protobuf": _package_version("protobuf"),
    }


def _model_artifact_hashes(model_dir: Path) -> dict[str, str]:
    allowed_suffixes = {".json", ".model", ".safetensors", ".bin", ".txt"}
    files = sorted(
        path
        for path in model_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    if not files or not (model_dir / "config.json").is_file():
        raise ValueError("local model directory is missing config/model/tokenizer files")
    weight_files = [
        path for path in files if path.suffix.lower() in {".safetensors", ".bin"}
    ]
    if not weight_files:
        raise ValueError("local model directory contains no model weight shards")
    return {
        path.name: sha256_file(path)
        for path in files
    }


def _validate_model_config(model_config: Any, sequence_length: int) -> None:
    expected = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    }
    for name, value in expected.items():
        observed = getattr(model_config, name, None)
        if observed != value:
            raise ValueError(
                f"formal trace requires model config {name}={value!r}, got {observed!r}"
            )
    if int(getattr(model_config, "max_position_embeddings", 0)) < sequence_length:
        raise ValueError("model max_position_embeddings is below 4096")


def _atomic_torch_save(torch: Any, payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_calibration_input(config: dict, tokenizer: Any) -> tuple[Any, dict]:
    from datasets import load_dataset

    calibration = config["calibration"]
    dataset = load_dataset(
        calibration["dataset"],
        calibration["subset"],
        split=calibration["split"],
        revision=calibration["revision"],
    )
    texts = list(dataset["text"])
    corpus = build_corpus(texts, calibration["corpus_joiner"])
    corpus_ids = tokenizer(
        corpus,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("Llama tokenizer has no bos_token_id")
    input_ids, start = select_prefill_tokens(
        corpus_ids,
        bos_token_id=int(bos_token_id),
        seed=int(calibration["seed"]),
        sampled_tokens=int(calibration["sampled_tokens"]),
    )
    if len(input_ids) != int(config["sequence_length"]):
        raise ValueError("calibration token selection did not produce 4096 tokens")
    metadata = {
        "dataset": {
            "dataset": calibration["dataset"],
            "subset": calibration["subset"],
            "split": calibration["split"],
            "revision": calibration["revision"],
            "fingerprint": str(getattr(dataset, "_fingerprint", "")),
            "corpus_joiner": calibration["corpus_joiner"],
            "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
            "row_count": len(texts),
        },
        "tokenization": {
            "seed": int(calibration["seed"]),
            "start_index": start,
            "sampled_tokens": int(calibration["sampled_tokens"]),
            "prepend_bos": True,
            "bos_token_id": int(bos_token_id),
            "corpus_token_count": len(corpus_ids),
            "input_ids": input_ids,
            "input_ids_sha256": token_ids_sha256(input_ids),
            "attention_mask_all_ones": True,
        },
    }
    return input_ids, metadata


def collect_trace(
    model_path: str | Path,
    output_dir: str | Path,
    config: dict,
    *,
    device: str | None = None,
) -> Path:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    layers, projections, sequence_length, expected_shape = validate_trace_config(config)
    model_dir = Path(model_path).expanduser()
    if not model_dir.is_absolute() or not model_dir.is_dir():
        raise ValueError("--model must be an existing absolute local model directory")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing trace directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.with_name(
        f".{output_dir.name}.collecting-{uuid.uuid4().hex}"
    )
    temporary_dir.mkdir()

    device_name = device or str(config.get("device", "cuda:0"))
    device_object = torch.device(device_name)
    if device_object.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal trace collection requires an available CUDA GPU")
    torch.cuda.set_device(device_object)

    model_config = AutoConfig.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_model_config(model_config, sequence_length)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    input_ids_list, calibration_metadata = _load_calibration_input(config, tokenizer)
    input_ids = torch.tensor(
        [input_ids_list],
        dtype=torch.long,
        device=device_object,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    model_hashes = _model_artifact_hashes(model_dir)
    attention_implementation = str(
        config.get("attention_implementation", "sdpa")
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation=attention_implementation,
        device_map={"": device_name},
    )
    model.eval()
    model.config.use_cache = False
    _validate_model_config(model.config, sequence_length)

    handles = []
    captured: set[str] = set()
    sample_entries: list[dict] = []

    def make_hook(layer_index: int, projection: str):
        sample_id = f"layer_{layer_index:02d}_{projection}"

        def hook(module: Any, args: tuple[Any, ...]) -> None:
            if sample_id in captured:
                raise RuntimeError(f"projection hook executed more than once: {sample_id}")
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError(f"{sample_id}: projection input is not a tensor")
            activation = args[0].detach()
            weight = module.weight.detach()
            if activation.dtype != torch.float16:
                raise TypeError(
                    f"{sample_id}: expected native FP16 activation, got {activation.dtype}"
                )
            if weight.dtype != torch.float16:
                raise TypeError(
                    f"{sample_id}: expected native FP16 weight, got {weight.dtype}"
                )
            if tuple(activation.shape) != (1, sequence_length, expected_shape[1]):
                raise ValueError(
                    f"{sample_id}: unexpected activation shape {tuple(activation.shape)}"
                )
            if tuple(weight.shape) != expected_shape:
                raise ValueError(
                    f"{sample_id}: unexpected weight shape {tuple(weight.shape)}"
                )

            activation_cpu = activation[0].to(device="cpu").contiguous()
            weight_cpu = weight.to(device="cpu").contiguous()
            if not torch.isfinite(activation_cpu).all():
                raise ValueError(f"{sample_id}: activation contains NaN/Inf")
            if not torch.isfinite(weight_cpu).all():
                raise ValueError(f"{sample_id}: weight contains NaN/Inf")
            path = temporary_dir / f"{sample_id}.pt"
            _atomic_torch_save(
                torch,
                {
                    "sample_id": sample_id,
                    "layer": layer_index,
                    "projection": projection,
                    "activation_fp16": activation_cpu,
                    "weight_fp16": weight_cpu,
                },
                path,
            )
            sample_entries.append(
                {
                    "sample_id": sample_id,
                    "file": path.name,
                    "layer": layer_index,
                    "projection": projection,
                    "sha256": sha256_file(path),
                    "activation": {
                        "shape": list(activation_cpu.shape),
                        "dtype": str(activation_cpu.dtype),
                    },
                    "weight": {
                        "shape": list(weight_cpu.shape),
                        "dtype": str(weight_cpu.dtype),
                    },
                }
            )
            captured.add(sample_id)

        return hook

    try:
        for layer_index in layers:
            attention = model.model.layers[layer_index].self_attn
            for projection in projections:
                module = getattr(attention, projection)
                if tuple(module.weight.shape) != expected_shape:
                    raise ValueError(
                        f"layer {layer_index} {projection} weight is "
                        f"{tuple(module.weight.shape)}, expected {expected_shape}"
                    )
                handles.append(
                    module.register_forward_pre_hook(
                        make_hook(layer_index, projection)
                    )
                )

        with torch.inference_mode():
            _ = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=False,
            )
        torch.cuda.synchronize(device_object)
    finally:
        for handle in handles:
            handle.remove()

    expected_ids = {
        f"layer_{layer:02d}_{projection}"
        for layer in layers
        for projection in projections
    }
    if captured != expected_ids:
        raise RuntimeError(
            f"trace hook set mismatch; missing={sorted(expected_ids - captured)}, "
            f"extra={sorted(captured - expected_ids)}"
        )

    manifest = {
        "version": RAW_TRACE_VERSION,
        "format": RAW_TRACE_FORMAT,
        "trace": {
            "batch_size": 1,
            "sequence_length": sequence_length,
            "valid_tokens": sequence_length,
            "dtype": "fp16",
            "layers": layers,
            "projections": projections,
            "expected_shape": list(expected_shape),
        },
        **calibration_metadata,
        "model": {
            "configured_name": config.get("model"),
            "local_directory_name": model_dir.name,
            "model_type": model.config.model_type,
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "max_position_embeddings": int(model.config.max_position_embeddings),
            "artifact_sha256": model_hashes,
        },
        "runtime": _runtime_metadata(torch, device_object),
        "inference": {
            "dtype": "fp16",
            "attention_implementation": attention_implementation,
            "use_cache": False,
            "local_files_only": True,
            "base_model_only": True,
        },
        "environment_source": dict(config.get("environment_source", {})),
        "samples": sorted(
            sample_entries,
            key=lambda entry: (entry["layer"], projections.index(entry["projection"])),
        ),
    }
    _write_json_atomic(manifest, temporary_dir / RAW_MANIFEST_NAME)
    validate_raw_trace(temporary_dir, config, deep=True)
    temporary_dir.rename(output_dir)
    return output_dir

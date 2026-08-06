"""Formal interleaved RTX 5090 experiment runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from ..ops.dispatch import benchmark_variant
from ..ops.extension import require_native
from ..trace.storage import load_prepared, sha256_file, validate_manifest
from .metadata import write_environment
from .metrics import mse_fp64, summarize_samples


def _throughput_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / ms / 1.0e9 if ms > 0 else 0.0


def _conversion_bytes(variant: str, stage: str, m: int, n: int, k: int) -> int:
    sizes = {
        ("o0", "weight_conversion"): n * k // 2 + n * (k // 32) + 2 * n * k,
        ("o0", "activation_conversion"): m * k + 4 * m + 2 * m * k,
        ("o1", "weight_conversion"): n * k // 2 + n * k,
        ("o2", "activation_conversion"): m * k + 4 * m + m * k // 2 + m * (k // 32),
    }
    return sizes.get((variant, stage), 0)


def _summarize_timings(timings: dict[str, list[float]], max_cv: float) -> tuple[dict, bool]:
    summaries = {stage: summarize_samples(values) for stage, values in timings.items() if values}
    stable = all(float(item["cv_percent"]) < max_cv for item in summaries.values())
    return summaries, stable


def _validate_benchmark_payload(payload: dict, repeats: int) -> None:
    if "output" not in payload or "timings_ms" not in payload:
        raise RuntimeError("native benchmark must return output and timings_ms")
    allowed = {"weight_conversion", "activation_conversion", "gemm", "total"}
    unknown = set(payload["timings_ms"]) - allowed
    if unknown:
        raise RuntimeError(f"unknown native timing stages: {sorted(unknown)}")
    for name, values in payload["timings_ms"].items():
        if len(values) != repeats:
            raise RuntimeError(f"stage {name} returned {len(values)} samples, expected {repeats}")


def run_experiment(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_limit: int | None = None,
    warmup_override: int | None = None,
    repeats_override: int | None = None,
) -> Path:
    import torch

    require_native()
    config_path, data_dir, output_dir = Path(config_path), Path(data_dir), Path(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config["experiment"]["formal"] or config["backend"]["required"] != "native":
        raise ValueError("formal runner requires experiment.formal=true and backend.required=native")
    timing = config["timing"]
    warmup = int(warmup_override if warmup_override is not None else timing["warmup"])
    repeats = int(repeats_override if repeats_override is not None else timing["repeats"])
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, formal=True)
    matrix = config["matrix"]
    configured_shape = [int(matrix["m"]), int(matrix["n"]), int(matrix["k"])]
    if manifest["matrix_shape"] != configured_shape:
        raise ValueError("experiment config and prepared manifest shapes do not match")
    for sample in manifest["samples"]:
        sample_path = data_dir / sample["file"]
        if not sample_path.is_file():
            raise FileNotFoundError(f"prepared sample is missing: {sample_path}")
        actual_sha = sha256_file(sample_path)
        if actual_sha != sample["sha256"]:
            raise ValueError(
                f"prepared sample checksum mismatch: {sample['sample_id']} "
                f"expected {sample['sha256']}, got {actual_sha}"
            )
    if sample_limit is not None and not 1 <= sample_limit <= len(manifest["samples"]):
        raise ValueError(f"samples must be in [1,{len(manifest['samples'])}]")
    samples = manifest["samples"] if sample_limit is None else manifest["samples"][:sample_limit]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = dict(config)
    resolved["timing"] = {**timing, "warmup": warmup, "repeats": repeats}
    (output_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    write_environment(output_dir / "environment.json")

    o0_output_dir = output_dir / "o0_outputs"
    o0_output_dir.mkdir()
    results_path = output_dir / "results.jsonl"
    max_cv = float(timing["max_cv_percent"])
    modes = list(config["modes"])
    variants = list(config["variants"])

    with results_path.open("w", encoding="utf-8") as stream:
        for sample in samples:
            inputs = load_prepared(data_dir / sample["file"], device="cuda")
            records: list[tuple[dict, object]] = []
            # Interleave variants inside each mode to reduce systematic temperature/boost drift.
            for mode in modes:
                for variant in variants:
                    payload = benchmark_variant(inputs, variant, mode, warmup, repeats, backend="native")
                    _validate_benchmark_payload(payload, repeats)
                    summaries, stable = _summarize_timings(payload["timings_ms"], max_cv)
                    m, n, k = inputs.shape
                    for stage, stats in summaries.items():
                        byte_count = _conversion_bytes(variant, stage, m, n, k)
                        stats["bytes_moved"] = byte_count
                        stats["throughput_gbps"] = (
                            byte_count / float(stats["median_ms"]) / 1.0e6
                            if byte_count and float(stats["median_ms"]) > 0
                            else 0.0
                        )
                    gemm_median = summaries.get("gemm", {}).get("median_ms", 0.0)
                    record = {
                        "schema_version": 1,
                        "sample_id": inputs.sample_id,
                        "variant": variant,
                        "mode": mode,
                        "shape": [m, n, k],
                        "timings_ms": payload["timings_ms"],
                        "summary": summaries,
                        "equivalent_tflops": _throughput_tflops(m, n, k, float(gemm_median)),
                        "mse_vs_o0": None,
                        "kernel": payload.get("kernel"),
                        "stable_cv": stable,
                        "valid": stable,
                    }
                    records.append((record, payload["output"]))

            reference = next(output for record, output in records if record["variant"] == "o0" and record["mode"] == "compute_only")
            if not torch.isfinite(reference).all():
                raise RuntimeError(f"{inputs.sample_id}: O0 output contains NaN/Inf")
            torch.save(reference.detach().cpu(), o0_output_dir / f"{inputs.sample_id}.pt")
            variant_outputs = {
                record["variant"]: output
                for record, output in records
                if record["mode"] == "compute_only"
            }
            mse = {name: mse_fp64(output, reference) for name, output in variant_outputs.items()}
            for record, _ in records:
                record["mse_vs_o0"] = mse[record["variant"]]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
            del inputs, records, reference, variant_outputs
            torch.cuda.empty_cache()

    shutil.copy2(data_dir / "manifest.json", output_dir / "data_manifest.json")
    return output_dir

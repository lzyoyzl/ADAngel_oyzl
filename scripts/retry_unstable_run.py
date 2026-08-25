#!/usr/bin/env python3
"""Paired retry of unstable formal-run records without one-sided selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=5)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.max_attempts <= 0:
        raise SystemExit("max-attempts must be positive")

    import torch

    from adangel.benchmark.metrics import mse_fp64
    from adangel.benchmark.runner import (
        _conversion_bytes,
        _summarize_timings,
        _throughput_tflops,
        _validate_benchmark_payload,
    )
    from adangel.ops.dispatch import benchmark_variant
    from adangel.ops.extension import require_native
    from adangel.trace.storage import load_prepared, validate_manifest

    require_native()
    run_dir = args.run.resolve()
    data_dir = args.data.resolve()
    results_path = run_dir / "results.jsonl"
    backup_path = run_dir / "results.initial.jsonl"
    audit_path = run_dir / "retry_audit.json"
    attempts_path = run_dir / "retry_attempts.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"results file is missing: {results_path}")
    if backup_path.exists() or audit_path.exists() or attempts_path.exists():
        raise FileExistsError("retry artifacts already exist; refusing to overwrite them")

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, formal=True)
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_key = {
        (record["sample_id"], record["variant"], record["mode"]): record
        for record in records
    }
    if len(by_key) != len(records):
        raise ValueError("formal run contains duplicate sample/variant/mode records")

    variants = list(config["variants"])
    modes = list(config["modes"])
    timing = config["timing"]
    warmup = int(timing["warmup"])
    repeats = int(timing["repeats"])
    conversion_inner_repeats = int(timing["conversion_inner_repeats"])
    max_cv = float(timing["max_cv_percent"])
    sample_by_id = {sample["sample_id"]: sample for sample in manifest["samples"]}
    sample_order = {sample["sample_id"]: index for index, sample in enumerate(manifest["samples"])}
    mode_order = {mode: index for index, mode in enumerate(modes)}
    targets = sorted(
        {
            (record["sample_id"], record["mode"])
            for record in records
            if not bool(record.get("valid", False))
        },
        key=lambda item: (sample_order[item[0]], mode_order[item[1]]),
    )
    if not targets:
        print(json.dumps({"passed": True, "targets": 0, "message": "no retries needed"}))
        return 0

    replacements = {}
    accepted = []
    unresolved = []
    attempt_count = 0
    with attempts_path.open("x", encoding="utf-8") as attempt_stream:
        for target_index, (sample_id, mode) in enumerate(targets):
            sample = sample_by_id[sample_id]
            inputs = load_prepared(data_dir / sample["file"], device="cuda")
            reference_path = run_dir / "o0_outputs" / f"{sample_id}.pt"
            reference = torch.load(
                reference_path,
                map_location=inputs.A_int8.device,
                weights_only=True,
            )
            accepted_records = None
            for attempt in range(1, args.max_attempts + 1):
                order = list(variants)
                shift = (target_index + attempt - 1) % len(order)
                order = order[shift:] + order[:shift]
                if (target_index + attempt) & 1:
                    order.reverse()
                pair_records = {}
                for variant in order:
                    payload = benchmark_variant(
                        inputs,
                        variant,
                        mode,
                        warmup,
                        repeats,
                        backend="native",
                        conversion_inner_repeats=conversion_inner_repeats,
                    )
                    _validate_benchmark_payload(
                        payload,
                        repeats,
                        mode,
                        conversion_inner_repeats,
                    )
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
                    output = payload["output"]
                    if output.dtype != torch.float32 or not torch.isfinite(output).all():
                        raise RuntimeError(f"{sample_id} {variant} {mode}: output is not finite FP32")
                    gemm_median = float(summaries.get("gemm", {}).get("median_ms", 0.0))
                    pair_records[variant] = {
                        "schema_version": 2,
                        "sample_id": sample_id,
                        "variant": variant,
                        "mode": mode,
                        "shape": [m, n, k],
                        "timings_ms": payload["timings_ms"],
                        "summary": summaries,
                        "equivalent_tflops": _throughput_tflops(m, n, k, gemm_median),
                        "mse_vs_o0": mse_fp64(output, reference),
                        "kernel": payload.get("kernel"),
                        "timing_method": payload.get("timing_method"),
                        "stable_cv": stable,
                        "valid": stable,
                    }
                torch.cuda.synchronize()
                attempt_count += 1
                all_stable = all(pair_records[variant]["valid"] for variant in variants)
                attempt_stream.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "mode": mode,
                            "attempt": attempt,
                            "order": order,
                            "all_variants_stable": all_stable,
                            "records": [pair_records[variant] for variant in variants],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                attempt_stream.flush()
                print(
                    f"{sample_id} {mode} attempt={attempt} stable={all_stable}",
                    flush=True,
                )
                if all_stable:
                    accepted_records = pair_records
                    accepted.append(
                        {"sample_id": sample_id, "mode": mode, "attempt": attempt}
                    )
                    break
            if accepted_records is None:
                unresolved.append({"sample_id": sample_id, "mode": mode})
            else:
                for variant in variants:
                    replacements[(sample_id, variant, mode)] = accepted_records[variant]
            del inputs, reference
            torch.cuda.empty_cache()

    audit = {
        "schema_version": 1,
        "source_results": str(results_path),
        "source_results_sha256": sha256_file(results_path),
        "policy": "paired_first_all-stable_attempt",
        "variants_rerun_together": variants,
        "max_cv_percent": max_cv,
        "warmup": warmup,
        "repeats": repeats,
        "conversion_inner_repeats": conversion_inner_repeats,
        "targets": len(targets),
        "attempts": attempt_count,
        "accepted": accepted,
        "unresolved": unresolved,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if unresolved:
        print(json.dumps({"passed": False, **audit}, indent=2))
        return 1

    merged = [
        replacements.get(
            (record["sample_id"], record["variant"], record["mode"]),
            record,
        )
        for record in records
    ]
    if any(not bool(record.get("valid", False)) for record in merged):
        raise RuntimeError("paired retry left invalid records despite no unresolved targets")
    temporary_path = run_dir / "results.retry.tmp.jsonl"
    with temporary_path.open("x", encoding="utf-8") as stream:
        for record in merged:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    results_path.replace(backup_path)
    temporary_path.replace(results_path)
    print(json.dumps({"passed": True, **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

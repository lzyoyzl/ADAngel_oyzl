#!/usr/bin/env python3
"""Paired RTX 5090 A/B benchmark for the O1 shared- and register-partial kernels."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


IMPLEMENTATIONS = ("shared_partial", "register_64x32", "register_128x128")
MODES = ("compute_only", "cold", "steady_state")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20250805)
    parser.add_argument("--samples", type=int, default=0, help="0 uses all manifest samples")
    parser.add_argument(
        "--max-paired-retries",
        type=int,
        default=0,
        help=(
            "retry only unstable shared/register_64x32 sample-mode pairs; "
            "0 disables targeted retries"
        ),
    )
    return parser.parse_args()


def summarize(values):
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    return {
        "count": len(numbers),
        "mean_ms": mean,
        "median_ms": statistics.median(numbers),
        "cv_percent": 0.0 if mean == 0.0 else statistics.pstdev(numbers) / mean * 100.0,
        "min_ms": min(numbers),
        "max_ms": max(numbers),
    }


def bootstrap_mean_ci(values, *, resamples, seed):
    rng = random.Random(seed)
    values = list(values)
    estimates = []
    for _ in range(resamples):
        estimates.append(statistics.fmean(rng.choice(values) for _ in values))
    estimates.sort()
    lower = estimates[math.floor(0.025 * (resamples - 1))]
    upper = estimates[math.ceil(0.975 * (resamples - 1))]
    return [lower, upper]


def mse64(actual, reference):
    return float((actual.double() - reference.double()).square().mean().item())


def main() -> int:
    args = parse_args()
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.conversion_inner_repeats <= 1
        or args.max_paired_retries < 0
    ):
        raise SystemExit("invalid timing arguments")
    if args.bootstrap_resamples < 1000:
        raise SystemExit("bootstrap-resamples must be at least 1000")

    import torch

    from adangel.ops.extension import require_variant
    from adangel.trace.storage import load_prepared, validate_manifest

    native = require_variant("o1")
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, formal=args.samples == 0)
    samples = manifest["samples"]
    if args.samples:
        samples = samples[: args.samples]

    records = []
    outputs = {}
    o0_outputs = {}
    for sample_index, sample in enumerate(samples):
        inputs = load_prepared(args.data / sample["file"], device="cuda")
        # O0 is measured only once here to define MSE. Its latency is not part of this A/B run.
        o0 = native.benchmark(
            "o0",
            "compute_only",
            inputs.A_int8,
            inputs.A_scale,
            inputs.W_mxfp4,
            inputs.W_scale,
            1,
            1,
            args.conversion_inner_repeats,
        )
        torch.cuda.synchronize()
        o0_outputs[sample["sample_id"]] = o0["output"]

        for mode_index, mode in enumerate(MODES):
            order = list(IMPLEMENTATIONS)
            if (sample_index + mode_index) & 1:
                order.reverse()
            for implementation in order:
                payload = native._benchmark_o1_impl(
                    implementation,
                    mode,
                    inputs.A_int8,
                    inputs.A_scale,
                    inputs.W_mxfp4,
                    inputs.W_scale,
                    args.warmup,
                    args.repeats,
                    args.conversion_inner_repeats,
                )
                torch.cuda.synchronize()
                stage_summaries = {
                    stage: summarize(values)
                    for stage, values in dict(payload["timings_ms"]).items()
                }
                output = payload["output"]
                if output.dtype != torch.float32 or not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"{sample['sample_id']} {implementation} {mode}: output is not finite FP32"
                    )
                outputs[(sample["sample_id"], implementation)] = output
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "implementation": implementation,
                        "mode": mode,
                        "stages": stage_summaries,
                        "kernel": dict(payload["kernel"]),
                        "mse_vs_o0": mse64(output, o0_outputs[sample["sample_id"]]),
                        "stable": all(
                            stage["cv_percent"] < args.max_cv_percent
                            for stage in stage_summaries.values()
                        ),
                    }
                )

        baseline = outputs[(sample["sample_id"], "shared_partial")]
        for candidate in IMPLEMENTATIONS[1:]:
            actual = outputs[(sample["sample_id"], candidate)]
            torch.testing.assert_close(actual, baseline, rtol=1e-3, atol=1e-3)

    initial_by_key = {
        (record["sample_id"], record["implementation"], record["mode"]): record
        for record in records
    }
    replacements = {}
    retry_attempts = []
    accepted_retries = []
    unresolved_retries = []
    if args.max_paired_retries:
        primary_pair = ("shared_partial", "register_64x32")
        sample_by_id = {sample["sample_id"]: sample for sample in samples}
        unstable_targets = [
            (sample["sample_id"], mode)
            for sample in samples
            for mode in MODES
            if not all(
                initial_by_key[(sample["sample_id"], implementation, mode)]["stable"]
                for implementation in primary_pair
            )
        ]
        targets_by_sample = {}
        for sample_id, mode in unstable_targets:
            targets_by_sample.setdefault(sample_id, []).append(mode)

        for sample_index, sample in enumerate(samples):
            sample_id = sample["sample_id"]
            target_modes = targets_by_sample.get(sample_id, ())
            if not target_modes:
                continue
            inputs = load_prepared(args.data / sample_by_id[sample_id]["file"], device="cuda")
            for mode in target_modes:
                accepted = None
                mode_index = MODES.index(mode)
                for attempt in range(1, args.max_paired_retries + 1):
                    order = list(primary_pair)
                    if (sample_index + mode_index + attempt) & 1:
                        order.reverse()
                    pair_records = {}
                    pair_outputs = {}
                    for implementation in order:
                        payload = native._benchmark_o1_impl(
                            implementation,
                            mode,
                            inputs.A_int8,
                            inputs.A_scale,
                            inputs.W_mxfp4,
                            inputs.W_scale,
                            args.warmup,
                            args.repeats,
                            args.conversion_inner_repeats,
                        )
                        torch.cuda.synchronize()
                        stage_summaries = {
                            stage: summarize(values)
                            for stage, values in dict(payload["timings_ms"]).items()
                        }
                        output = payload["output"]
                        if output.dtype != torch.float32 or not torch.isfinite(output).all():
                            raise RuntimeError(
                                f"{sample_id} {implementation} {mode} retry {attempt}: "
                                "output is not finite FP32"
                            )
                        pair_outputs[implementation] = output
                        pair_records[implementation] = {
                            "sample_id": sample_id,
                            "implementation": implementation,
                            "mode": mode,
                            "stages": stage_summaries,
                            "kernel": dict(payload["kernel"]),
                            "mse_vs_o0": mse64(output, o0_outputs[sample_id]),
                            "stable": all(
                                stage["cv_percent"] < args.max_cv_percent
                                for stage in stage_summaries.values()
                            ),
                        }
                    torch.testing.assert_close(
                        pair_outputs["register_64x32"],
                        pair_outputs["shared_partial"],
                        rtol=1e-3,
                        atol=1e-3,
                    )
                    old_mse = pair_records["shared_partial"]["mse_vs_o0"]
                    new_mse = pair_records["register_64x32"]["mse_vs_o0"]
                    mse_passed = abs(new_mse - old_mse) <= 1e-12 + 1e-5 * abs(old_mse)
                    pair_stable = all(
                        pair_records[implementation]["stable"]
                        for implementation in primary_pair
                    )
                    retry_attempts.append(
                        {
                            "sample_id": sample_id,
                            "mode": mode,
                            "attempt": attempt,
                            "order": order,
                            "pair_stable": pair_stable,
                            "mse_regression_passed": mse_passed,
                            "records": [pair_records[key] for key in primary_pair],
                        }
                    )
                    if pair_stable and mse_passed:
                        accepted = pair_records
                        for implementation in primary_pair:
                            replacements[(sample_id, implementation, mode)] = pair_records[
                                implementation
                            ]
                        accepted_retries.append(
                            {"sample_id": sample_id, "mode": mode, "attempt": attempt}
                        )
                        break
                if accepted is None:
                    unresolved_retries.append({"sample_id": sample_id, "mode": mode})

        records = [
            replacements.get(
                (record["sample_id"], record["implementation"], record["mode"]),
                record,
            )
            for record in records
        ]

    by_key = {
        (record["sample_id"], record["implementation"], record["mode"]): record
        for record in records
    }
    comparisons = {}
    for candidate_index, candidate in enumerate(IMPLEMENTATIONS[1:]):
        by_mode = {}
        for mode_index, mode in enumerate(MODES):
            stage = "gemm" if mode == "compute_only" else "total"
            ratios = [
                by_key[(sample["sample_id"], "shared_partial", mode)]["stages"][stage][
                    "median_ms"
                ]
                / by_key[(sample["sample_id"], candidate, mode)]["stages"][stage]["median_ms"]
                for sample in samples
            ]
            by_mode[mode] = {
                "stage": stage,
                "paired_speedup_geomean": math.exp(
                    statistics.fmean(math.log(value) for value in ratios)
                ),
                "paired_speedup_mean": statistics.fmean(ratios),
                "paired_speedup_bootstrap_95ci": bootstrap_mean_ci(
                    ratios,
                    resamples=args.bootstrap_resamples,
                    seed=args.seed + candidate_index * len(MODES) + mode_index,
                ),
            }
        speedups = []
        mse_regressions = []
        for sample in samples:
            sample_id = sample["sample_id"]
            baseline_record = by_key[(sample_id, "shared_partial", "compute_only")]
            candidate_record = by_key[(sample_id, candidate, "compute_only")]
            speedups.append(
                baseline_record["stages"]["gemm"]["median_ms"]
                / candidate_record["stages"]["gemm"]["median_ms"]
            )
            old_mse = baseline_record["mse_vs_o0"]
            new_mse = candidate_record["mse_vs_o0"]
            mse_regressions.append(abs(new_mse - old_mse) <= 1e-12 + 1e-5 * abs(old_mse))
        ci = bootstrap_mean_ci(
            speedups,
            resamples=args.bootstrap_resamples,
            seed=args.seed + candidate_index,
        )
        all_stable = all(
            record["stable"]
            for record in records
            if record["implementation"] in {"shared_partial", candidate}
        )
        comparisons[candidate] = {
            "by_mode": by_mode,
            "paired_compute_speedup_geomean": math.exp(
                statistics.fmean(math.log(value) for value in speedups)
            ),
            "paired_compute_speedup_mean": statistics.fmean(speedups),
            "paired_compute_speedup_bootstrap_95ci": ci,
            "all_timing_stages_cv_below_threshold": all_stable,
            "mse_regression_passed": all(mse_regressions),
            "spill_audit_required": True,
            "eligible_except_spill_audit": (
                ci[0] > 1.0 and all_stable and all(mse_regressions)
            ),
        }

    report = {
        "schema_version": 1,
        "data": str(args.data),
        "samples": len(samples),
        "timing": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "conversion_inner_repeats": args.conversion_inner_repeats,
            "max_cv_percent": args.max_cv_percent,
            "ordering": "paired blocks; implementation order alternates by sample and mode",
        },
        "production_implementation_during_ab": "shared_partial",
        "targeted_paired_retry": {
            "enabled": bool(args.max_paired_retries),
            "max_attempts": args.max_paired_retries,
            "acceptance": (
                "both shared_partial and register_64x32 have every stage CV below "
                "threshold and pass MSE regression; latency is not an acceptance input"
            ),
            "attempts": retry_attempts,
            "accepted": accepted_retries,
            "unresolved": unresolved_retries,
        },
        "records": records,
        "comparisons": comparisons,
        "promotion_rule": (
            "correctness and MSE pass, every timed stage CV < threshold, paired speedup "
            "bootstrap 95% CI lower bound > 1, and separate SASS/resource audit reports no spill"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))
    return 1 if unresolved_retries else 0


if __name__ == "__main__":
    raise SystemExit(main())

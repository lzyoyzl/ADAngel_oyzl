#!/usr/bin/env python3
"""Paired RTX 5090 benchmark for O3/O4 optimization candidates."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


IMPLEMENTATIONS = {
    "o3": (
        "n16_k128",
        "n16_k256",
        "n32_k128",
        "n16_k128_dual",
        "n32_k256_dual",
        "n16_k128_swizzle",
        "n32_k128_swizzle",
        "m64_n16_k128",
        "m64_n32_k128",
        "m64_n16_k128_cute_ldsm",
        "n16_k128_cute_ldsm",
        "n32_k128_cute_ldsm",
        "n16_k128_ldsm_swizzle",
    ),
    "o4": (
        "n64_k256",
        "n64_k256_split2",
        "n64_k256_cache_b",
        "n64_k256_split2_cache_b",
        "m64_n64_k512",
        "m64_n64_k512_optimized",
    ),
}
MODES = ("compute_only", "cold", "steady_state")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("o3", "o4"), required=True)
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
        "--implementations",
        nargs="+",
        help=(
            "optional candidate subset; the variant baseline is always included first"
        ),
    )
    parser.add_argument(
        "--cv-policy", choices=("diagnostic", "strict"), default="diagnostic"
    )
    return parser.parse_args()


def summarize(values):
    numbers = [float(value) for value in values]
    mean = statistics.fmean(numbers)
    return {
        "count": len(numbers),
        "mean_ms": mean,
        "median_ms": statistics.median(numbers),
        "cv_percent": 0.0 if mean == 0 else statistics.pstdev(numbers) / mean * 100.0,
        "min_ms": min(numbers),
        "max_ms": max(numbers),
    }


def bootstrap_mean_ci(values, *, resamples, seed):
    values = list(values)
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(resamples)
    )
    return [
        estimates[math.floor(0.025 * (resamples - 1))],
        estimates[math.ceil(0.975 * (resamples - 1))],
    ]


def mse64(actual, reference):
    return float((actual.double() - reference.double()).square().mean().item())


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.warmup < 0 or args.bootstrap_resamples < 1000:
        raise SystemExit("invalid benchmark arguments")

    import torch

    from adangel.ops.extension import require_variant
    from adangel.trace.storage import load_prepared, validate_manifest

    native = require_variant(args.variant)
    benchmark = (
        native._benchmark_o3_impl if args.variant == "o3" else native._benchmark_o4_impl
    )
    available = IMPLEMENTATIONS[args.variant]
    baseline = available[0]
    if args.implementations:
        unknown = sorted(set(args.implementations) - set(available))
        if unknown:
            raise SystemExit(f"unknown {args.variant} implementations: {unknown}")
        implementations = (baseline,) + tuple(
            implementation
            for implementation in args.implementations
            if implementation != baseline
        )
        implementations = tuple(dict.fromkeys(implementations))
    else:
        implementations = available
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, formal=args.samples == 0)
    samples = manifest["samples"][: args.samples or None]
    records = []

    for sample_index, sample in enumerate(samples):
        inputs = load_prepared(args.data / sample["file"], device="cuda")
        o0 = native.benchmark(
            "o0", "compute_only", inputs.A_int8, inputs.A_scale,
            inputs.W_mxfp4, inputs.W_scale, 1, 1, args.conversion_inner_repeats,
        )
        torch.cuda.synchronize()
        o0_output = o0["output"]
        outputs = {}
        for mode_index, mode in enumerate(MODES):
            order = list(implementations)
            if (sample_index + mode_index) & 1:
                order.reverse()
            for implementation in order:
                payload = benchmark(
                    implementation, mode, inputs.A_int8, inputs.A_scale,
                    inputs.W_mxfp4_g128, inputs.W_scale_g128,
                    args.warmup, args.repeats, args.conversion_inner_repeats,
                )
                torch.cuda.synchronize()
                output = payload["output"]
                if output.dtype != torch.float32 or not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"{sample['sample_id']} {implementation} {mode}: non-finite FP32 output"
                    )
                stages = {
                    stage: summarize(values)
                    for stage, values in dict(payload["timings_ms"]).items()
                }
                outputs[(mode, implementation)] = output
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "implementation": implementation,
                        "mode": mode,
                        "stages": stages,
                        "kernel": dict(payload["kernel"]),
                        "mse_vs_o0": mse64(output, o0_output),
                        "stable": all(
                            stage["cv_percent"] < args.max_cv_percent
                            for stage in stages.values()
                        ),
                    }
                )
        for mode in MODES:
            baseline_output = outputs[(mode, baseline)]
            for implementation in implementations[1:]:
                torch.testing.assert_close(
                    outputs[(mode, implementation)], baseline_output, rtol=1e-3, atol=1e-3
                )

    by_key = {
        (record["sample_id"], record["implementation"], record["mode"]): record
        for record in records
    }
    comparisons = {}
    for candidate_index, candidate in enumerate(implementations[1:]):
        by_mode = {}
        for mode_index, mode in enumerate(MODES):
            stage = "gemm" if mode == "compute_only" else "total"
            speedups = [
                by_key[(sample["sample_id"], baseline, mode)]["stages"][stage]["median_ms"]
                / by_key[(sample["sample_id"], candidate, mode)]["stages"][stage]["median_ms"]
                for sample in samples
            ]
            by_mode[mode] = {
                "stage": stage,
                "paired_speedup_geomean": math.exp(
                    statistics.fmean(math.log(value) for value in speedups)
                ),
                "paired_speedup_mean": statistics.fmean(speedups),
                "paired_speedup_bootstrap_95ci": bootstrap_mean_ci(
                    speedups,
                    resamples=args.bootstrap_resamples,
                    seed=args.seed + 17 * candidate_index + mode_index,
                ),
            }
        mse_passed = all(
            abs(
                by_key[(sample["sample_id"], candidate, "compute_only")]["mse_vs_o0"]
                - by_key[(sample["sample_id"], baseline, "compute_only")]["mse_vs_o0"]
            )
            <= 1e-12
            + 1e-5
            * abs(by_key[(sample["sample_id"], baseline, "compute_only")]["mse_vs_o0"])
            for sample in samples
        )
        all_stable = all(
            record["stable"]
            for record in records
            if record["implementation"] in {baseline, candidate}
        )
        compute_ci = by_mode["compute_only"]["paired_speedup_bootstrap_95ci"]
        performance_eligible = compute_ci[0] > 1.0 and mse_passed
        comparisons[candidate] = {
            "by_mode": by_mode,
            "mse_regression_passed": mse_passed,
            "all_timing_stages_cv_below_threshold": all_stable,
            "performance_eligible_except_spill_audit": performance_eligible,
            "strict_eligible_except_spill_audit": performance_eligible and all_stable,
            "eligible_except_spill_audit": performance_eligible
            and (all_stable or args.cv_policy == "diagnostic"),
        }

    report = {
        "schema_version": 1,
        "variant": args.variant,
        "baseline": baseline,
        "data": str(args.data),
        "samples": len(samples),
        "implementations": implementations,
        "timing": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "conversion_inner_repeats": args.conversion_inner_repeats,
            "max_cv_percent": args.max_cv_percent,
            "cv_policy": args.cv_policy,
            "ordering": "candidate order alternates by sample and mode",
        },
        "records": records,
        "comparisons": comparisons,
        "promotion_rule": (
            "correctness and MSE pass; paired compute speedup bootstrap 95% CI lower bound "
            "> 1; same-entry TMA+MMA and zero-spill resource audit pass"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

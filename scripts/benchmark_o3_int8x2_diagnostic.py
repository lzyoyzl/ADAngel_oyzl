#!/usr/bin/env python3
"""Compare formal O3 with the exact Split diagnostic that uses two INT8 MMA paths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MODES = ("conversion_only", "compute_only", "cold", "steady_state")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=0, help="0 uses all samples")
    parser.add_argument("--max-cv-percent", type=float, default=3.0)
    return parser.parse_args()


def summarize(values):
    values = [float(value) for value in values]
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "cv_percent": 0.0 if mean == 0 else statistics.pstdev(values) / mean * 100.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def mse64(actual, reference):
    return float((actual.double() - reference.double()).square().mean().item())


def main() -> int:
    args = arguments()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("invalid timing arguments")

    import torch

    from adangel.ops.extension import require_variant
    from adangel.trace.storage import load_prepared, validate_manifest

    native = require_variant("o3")
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, formal=args.samples == 0, require_arbitrary_bits=True)
    samples = manifest["samples"][: args.samples or None]
    records = []

    for sample_index, sample in enumerate(samples):
        inputs = load_prepared(args.data / sample["file"], device="cuda")
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
        o0_output = o0["output"]
        for mode_index, mode in enumerate(MODES):
            order = ["formal_o3", "int8x2"]
            if (sample_index + mode_index) & 1:
                order.reverse()
            outputs = {}
            for implementation in order:
                if implementation == "formal_o3":
                    payload = native._benchmark_o3_impl(
                        "production",
                        mode,
                        inputs.A_int8,
                        inputs.A_scale,
                        inputs.W_mxfp4_g128,
                        inputs.W_scale_g128,
                        args.warmup,
                        args.repeats,
                        args.conversion_inner_repeats,
                    )
                else:
                    payload = native._benchmark_o3_split_int8x2(
                        mode,
                        inputs.A_int8,
                        inputs.A_scale,
                        inputs.W_mxfp4_g128,
                        inputs.W_scale_g128,
                        args.warmup,
                        args.repeats,
                        args.conversion_inner_repeats,
                    )
                torch.cuda.synchronize()
                output = payload["output"]
                if output.dtype != torch.float32 or not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"{sample['sample_id']} {implementation} {mode}: invalid output"
                    )
                stages = {
                    name: summarize(values)
                    for name, values in dict(payload["timings_ms"]).items()
                }
                outputs[implementation] = output
                records.append(
                    {
                        "sample_id": sample["sample_id"],
                        "implementation": implementation,
                        "mode": mode,
                        "stages": stages,
                        "mse_vs_o0": mse64(output, o0_output),
                        "kernel": dict(payload["kernel"]),
                        "stable": all(
                            stage["cv_percent"] < args.max_cv_percent
                            for stage in stages.values()
                        ),
                    }
                )
            torch.testing.assert_close(
                outputs["int8x2"], outputs["formal_o3"], rtol=1e-3, atol=1e-3
            )

    by_key = {
        (record["sample_id"], record["implementation"], record["mode"]): record
        for record in records
    }
    summary = {}
    for implementation in ("formal_o3", "int8x2"):
        by_mode = {}
        for mode in MODES:
            stage = "total" if mode != "compute_only" else "gemm"
            values = [
                by_key[(sample["sample_id"], implementation, mode)]["stages"][stage][
                    "median_ms"
                ]
                for sample in samples
            ]
            by_mode[mode] = {
                "reported_stage": stage,
                "median_of_sample_medians_ms": statistics.median(values),
                "mean_of_sample_medians_ms": statistics.fmean(values),
            }
        mse_values = [
            by_key[(sample["sample_id"], implementation, "compute_only")]["mse_vs_o0"]
            for sample in samples
        ]
        summary[implementation] = {
            "modes": by_mode,
            "mse_vs_o0_median": statistics.median(mse_values),
            "mse_vs_o0_mean": statistics.fmean(mse_values),
        }

    speedups = []
    for sample in samples:
        formal = by_key[(sample["sample_id"], "formal_o3", "compute_only")]["stages"][
            "gemm"
        ]["median_ms"]
        diagnostic = by_key[(sample["sample_id"], "int8x2", "compute_only")][
            "stages"
        ]["gemm"]["median_ms"]
        speedups.append(formal / diagnostic)
    summary["comparison"] = {
        "formal_o3_over_int8x2_paired_speedup_median": statistics.median(speedups),
        "formal_o3_over_int8x2_paired_speedup_mean": statistics.fmean(speedups),
        "all_records_stable": all(record["stable"] for record in records),
        "interpretation": (
            "INT8x2 is diagnostic only: it preserves O3 arithmetic but violates the "
            "required U4xS4/S4xS4 Tensor Core interface"
        ),
    }

    result = {
        "schema_version": 1,
        "data": str(args.data),
        "samples": len(samples),
        "timing": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "conversion_inner_repeats": args.conversion_inner_repeats,
            "max_cv_percent": args.max_cv_percent,
            "ordering": "formal O3 and INT8x2 alternate by sample and mode",
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

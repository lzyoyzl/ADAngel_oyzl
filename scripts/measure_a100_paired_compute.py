#!/usr/bin/env python3
"""Supplement the full run with tightly interleaved, unfiltered O1/O3 timing."""
import argparse
import json
from pathlib import Path
import statistics
import time

from run_a100_experiment import command, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    if args.output.exists() or min(args.rounds, args.repeats) < 1 or args.warmup < 0:
        raise SystemExit("Use a fresh output directory and valid repetition counts")
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared, validate_manifest, sha256_file
    if torch.cuda.get_device_capability() != (8, 0):
        raise SystemExit("SM80 GPU required")
    manifest = json.loads((args.data / "manifest.json").read_text())
    validate_manifest(manifest, formal=True, require_arbitrary_bits=True)
    args.output.mkdir(parents=True)
    env = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               cuda=torch.version.cuda, git_commit=command("git", "rev-parse", "HEAD"),
               binary_sha256=sha256_file(Path(native.__file__)),
               rounds=args.rounds, repeats_per_round=args.repeats,
               warmup_per_call=args.warmup, mode="compute_only",
               policy="Alternating O1/O3 order each round; no samples removed; unlocked clocks")
    (args.output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    summaries = []
    for i, sample in enumerate(manifest["samples"]):
        path = args.data / sample["file"]
        assert sha256_file(path) == sample["sha256"]
        x = load_prepared(path, device="cuda")
        o0 = native.benchmark_o0(x.A_int8, x.A_scale, x.W_mxfp4, x.W_scale,
                                 "compute_only", 5, 10, 100)["output"]
        snapshot = dict(time_unix=time.time(), sample_id=x.sample_id,
                        processes=command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                          "--format=csv"),
                        gpu=command("nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu",
                                    "--format=csv"))
        with (args.output / "gpu_snapshots.jsonl").open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
        timings = {v: [] for v in ("o1", "o3")}
        outputs = {}
        ratios = []
        for r in range(args.rounds):
            medians = {}
            for v in (("o1", "o3") if (i + r) % 2 == 0 else ("o3", "o1")):
                w, ws = (x.W_mxfp4, x.W_scale) if v == "o1" else (x.W_mxfp4_g128, x.W_scale_g128)
                p = native.benchmark(v, "compute_only", x.A_int8, x.A_scale, w, ws,
                                     args.warmup, args.repeats, 100)
                values = list(p["timings_ms"]["gemm"])
                timings[v].extend(values)
                medians[v] = statistics.median(values)
                outputs[v] = p["output"]
                record = dict(sample_id=x.sample_id, variant=v, round=r, timings_ms=values,
                              summary=stats(values), kernel=dict(p["kernel"]))
                with (args.output / "rounds.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
            ratios.append(medians["o1"] / medians["o3"])
        mses = {}
        for v, y in outputs.items():
            assert y.dtype == torch.float32 and torch.isfinite(y).all()
            mses[v] = float((y.double() - o0.double()).square().mean())
        summaries.append(dict(sample_id=x.sample_id, timings={v: stats(t) for v, t in timings.items()},
                              mse_vs_o0=mses, round_ratios=ratios,
                              paired_ratio_median=statistics.median(ratios)))
        print(f"Completed {i+1}/24 {x.sample_id}: ratio={statistics.median(ratios):.4f}", flush=True)
    result = dict(samples=summaries,
                  o3_throughput_over_o1=statistics.median(s["paired_ratio_median"] for s in summaries),
                  latency_medians_ms={v: statistics.median(s["timings"][v]["median_ms"] for s in summaries)
                                      for v in ("o1", "o3")},
                  warning="Concurrent load, if present, is retained; this is not an exclusive-GPU benchmark")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()

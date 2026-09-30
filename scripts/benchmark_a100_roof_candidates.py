#!/usr/bin/env python3
"""Synthetic screen of internal O3/O7/O8 candidates; NOT formal trace acceptance.

Each candidate must equal the unchanged production arithmetic bitwise. Timing
is paired in alternating orders, with raw samples and no outlier removal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true", required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tunes", type=int, nargs="+", default=[-1, 0, 1, 2, 3])
    p.add_argument("--variants", nargs="+", choices=["o3", "o7", "o8"], default=["o3", "o7", "o8"])
    args = p.parse_args()
    if args.output.exists() or args.size < 256 or args.size % 256 or args.warmup < 0 or min(args.repeats, args.rounds) < 1 or any(t not in (-1,0,1,2,3,6,7) for t in args.tunes):
        p.error("fresh output, tile alignment and valid repetitions/tunes required")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4, reference
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("SM80 required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)

    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

    def append(name, obj):
        with (args.output / name).open("a") as out:
            out.write(json.dumps(obj, allow_nan=False) + "\n")

    save("environment.json", {"git_commit": command("git", "rev-parse", "HEAD"),
         "binary": native.__file__, "binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
         "device": torch.cuda.get_device_name(), "torch": torch.__version__,
         "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "scope": "synthetic_prepared_core_only_not_real_trace_or_full_timing_acceptance"})

    def inputs(variant, m, n, k, pattern="random"):
        torch.manual_seed(930 + k)
        a = torch.randint(-128 if variant != "o8" else -32, 128 if variant != "o8" else 32,
                          (m, k), device="cuda", dtype=torch.int8)
        if pattern == "zero":
            a.zero_()
        if pattern == "extrema":
            a[:, ::2] = -128 if variant != "o8" else -32
            a[:, 1::2] = 127 if variant != "o8" else 31
        if variant == "o3":
            mx = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
            asc = torch.linspace(.001, .03, m, device="cuda")
            codes = ((torch.arange(n * (k // 128), device="cuda").reshape(n, -1) * 7) % 13 + 116).byte()
            if pattern == "zero_scale":
                asc.zero_()
            def baseline(warmup, repeats):
                return native.benchmark("o3", "compute_only", a, asc, mx, codes, warmup, repeats, 100, "production")
            b = baseline(0, 1)
            return (b["converted_activation"], asc, b["converted_weight"], codes), b["output"], baseline
        w = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
        if pattern == "extrema":
            w[:, ::2], w[:, 1::2] = -8, 7
        def scales(rows, multiplier):
            r = torch.arange(rows, device="cuda")[:, None]
            g = torch.arange(k // 128, device="cuda")[None, :]
            return ((1 + (r * multiplier + g * 29) % 113 / 128) * torch.exp2(((r + 3*g) % 7 - 10).float())).contiguous()
        asc, wsc = scales(m, 13), scales(n, 17)
        if pattern == "zero_scale":
            asc[:, ::2], wsc[:, 1::2] = 0, 0
        packed_a, packed_w = split_int8_to_packed_int4(a), pack_q4(w)
        def baseline(warmup, repeats):
            return native._benchmark_split_grouped(packed_a, asc, packed_w, wsc, warmup, repeats, "64x128x256")
        b = baseline(0, 1)
        if m <= 128 and n <= 256:
            torch.testing.assert_close(b["output"].cpu(), reference(a, asc, w, wsc), rtol=1e-3, atol=1e-3)
        # Conversion/reordering is explicitly outside these compute-only events.
        return (packed_a, asc.T.contiguous().T, packed_w, wsc.T.contiguous().T), b["output"], baseline

    if args.validate:
        checks = []
        for variant in args.variants:
            for m, n, k in ((64, 128, 256), (128, 256, 512), (64, 128, 768), (128, 128, 4096)):
                for pattern in ("random", "zero", "extrema", "zero_scale"):
                    values, expected, _ = inputs(variant, m, n, k, pattern)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for tune in args.tunes:
                            result = native._benchmark_roof_candidate(variant, tune, *values, 0, 1)
                            stream.synchronize()
                            y = result["output"]
                            assert torch.isfinite(y).all() and torch.equal(y.view(torch.int32), expected.view(torch.int32)), (variant, tune, m, n, k, pattern)
                            checks.append(dict(variant=variant, tune=tune, shape=[m,n,k], pattern=pattern,
                                               bitwise_equal=True, mse_vs_production=0.0, kernel=dict(result["kernel"])))
        save("validation.json", {"passed": True, "checks": checks})
        print("synthetic bitwise checks:", len(checks), flush=True)

    records = []
    for variant in args.variants:
        values, expected, baseline = inputs(variant, args.size, args.size, args.size)
        for r in range(args.rounds):
            append("gpu_snapshots.jsonl", {"variant": variant, "round": r, "time": time.time(),
                   "gpu": command("nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv")})
            order = args.tunes[r % len(args.tunes):] + args.tunes[:r % len(args.tunes)]
            if r % 2:
                order.reverse()
            for tune in order:
                result = native._benchmark_roof_candidate(variant, tune, *values, args.warmup, args.repeats)
                y = result["output"]
                assert torch.isfinite(y).all() and torch.equal(y.view(torch.int32), expected.view(torch.int32)), (variant, tune, "large")
                raw = list(result["gemm_ms"])
                row = dict(variant=variant, tune=tune, round=r, raw_ms=raw, summary=stats(raw),
                           kernel=dict(result["kernel"]), bitwise_equal_production=True, mse_vs_production=0.0)
                records.append(row);append("results.jsonl", row)
                print(variant, r, tune, row["summary"], flush=True)
    summary = []
    for v in args.variants:
        for tune in args.tunes:
            group = [r for r in records if r["variant"] == v and r["tune"] == tune]
            summary.append(dict(variant=v, tune=tune, median_ms=statistics.median(r["summary"]["median_ms"] for r in group),
                                cv_failed_rounds=sum(r["summary"]["cv_percent"] >= 3 for r in group)))
    save("summary.json", {"scope": "synthetic_screen_only", "records": summary})


if __name__ == "__main__":
    main()

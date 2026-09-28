#!/usr/bin/env python3
"""Paired SYNTHETIC O0/O3/O5/O6 benchmark; never a real-trace acceptance run.

Source-format preparation is outside timing. This entry deliberately has no
implicit data directory: pending provenance choices cannot silently change a
formal experiment. Output folders are immutable once created.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats

TILES = ("64x64x128", "64x128x256")
MODES = ("conversion_only", "compute_only", "cold", "steady_state")


def conversion_bytes(case, stage, m, n, k):
    """Logical unique tensor bytes read + written, NOT measured DRAM traffic.

    Scalar metadata counted once; cache lines/redundant thread accesses excluded.
    FP6 source is byte-padded, not a densely packed six-bit array.
    """
    variant = case.split("/")[0]
    if variant == "o0":
        w, a = n * k // 2 + n * k // 32 + 2 * n * k, 3 * m * k + 4 * m
    elif variant == "o3":
        w, a = n * k, 2 * m * k
    elif variant == "o5":
        w = n * k + 5 * n * (k // 128) + 4
        a = 2 * m * k + 5 * m * (k // 128)
    elif variant == "o6":
        # HiF4: scale + 2-byte micro8 + 4-byte micro4; output scale is FP32.
        w = n * k + 11 * n * (k // 128)
        a = 2 * m * k + 5 * m * (k // 128) + 4
    else:
        raise ValueError("unknown variant")
    return {"weight_conversion": w, "activation_conversion": a, "total": w + a}.get(stage, 0)


def integer_reference(a, asc, w, wsc):
    """Independent FP64 exact integer group dots and FP32 ordered accumulation.

    For this bounded synthetic corpus, FP64 has sufficient precision to emulate
    each FP32 FMA. Not an arbitrary-exponent IEEE-754 emulator.
    """
    import torch
    y = torch.zeros((a.shape[0], w.shape[0]), device=a.device, dtype=torch.float32)
    for group in range(a.shape[1] // 128):
        sl = slice(group * 128, (group + 1) * 128)
        partial = a[:, sl].double() @ w[:, sl].double().T
        scale = (asc[:, group, None] * wsc[None, :, group]).float()
        y = (partial * scale.double() + y.double()).float()
    return y


def summarize_records(records):
    """Pair by sample/round/mode, never treat event repeats as new samples."""
    summary = []
    index = {(r["sample_id"], r["round"], r["mode"], r["case"]): r for r in records}
    if len(index) != len(records):
        raise ValueError("duplicate measurement key")
    for case, mode in sorted({(r["case"], r["mode"]) for r in records}):
        selected = [r for r in records if r["case"] == case and r["mode"] == mode]
        stage = "gemm" if mode == "compute_only" else "total"
        ratios = []
        for r in selected:
            o0 = index[(r["sample_id"], r["round"], mode, "o0")]
            ratios.append(o0["summary"][stage]["median_ms"] / r["summary"][stage]["median_ms"])
        # O3 historically sums separately amortized conversion samples. Do not
        # label its conversion total as a joint-event speedup over O0.
        comparable = not (case == "o3" and mode == "conversion_only")
        summary.append({"case": case, "mode": mode, "records": len(selected),
                        "stage": stage, "median_ms": statistics.median(r["summary"][stage]["median_ms"] for r in selected),
                        "mean_ms": statistics.fmean(r["summary"][stage]["mean_ms"] for r in selected),
                        "paired_speedup_vs_o0_median": statistics.median(ratios) if comparable else None,
                        "pairing_scope": "synthetic_input_round_not_real_samples",
                        "median_mse_vs_o0": statistics.median(r["mse_vs_o0"] for r in selected),
                        "max_stage_cv_percent": max(s["cv_percent"] for r in selected for s in r["summary"].values()),
                        "cv_failed_records": sum(not r["timing_stable_cv3"] for r in selected),
                        "total_timing": sorted({r["total_timing"] for r in selected})})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", required=True,
                        help="explicitly acknowledge this is NOT a real-trace run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--tiles", nargs="+", choices=TILES, default=list(TILES))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output directory")
    if args.size < 512 or args.size % 256 or args.warmup < 0 or args.repeats < 2 or args.inner < 2 or args.rounds < 1:
        parser.error("size must be >=512 and divisible by256; invalid repetition count")
    if len(set(args.modes)) != len(args.modes) or len(set(args.tiles)) != len(args.tiles):
        parser.error("duplicate mode/tile")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from adangel.trace.storage import sha256_file
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires A100 SM80")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]

    def save(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    def append(name, data):
        with (args.output / name).open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False) + "\n")

    def snapshot():
        return {"time": time.time(), "gpu": command("nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv"),
                "processes": command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv")}

    save("environment.json", {"git_commit": command("git", "rev-parse", "HEAD"),
         "binary_sha256": sha256_file(Path(native.__file__)), "torch": torch.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
         "cuda_sources_sha256": {name: sha256_file(root / name) for name in
             ("csrc/sm80/o1_o3.cu", "csrc/sm80/o3_optimized.cuh", "csrc/sm80/split_grouped.cuh",
              "csrc/sm80/mixed_conversion.cuh", "csrc/sm80/mixed_benchmark.cuh")},
         "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "scope": "synthetic_source_formats_not_real_trace", "formal_o5_o6_complete": False,
         "policy": "No filtering/retry-until-pass; same-process cyclic/reversed order; unlocked/shared GPU; conversion amortized, end-to-end direct"})
    gen = torch.Generator().manual_seed(args.seed)
    acpu = (torch.randn((args.size, args.size), generator=gen) * .5).half()
    wcpu = (torch.randn((args.size, args.size), generator=gen) * .125).half()
    save("input_manifest.json", {"sample_id": f"synthetic_{args.seed}", "shape": [args.size] * 3,
         "origin": "synthetic_fp16_gaussian", "seed": args.seed,
         "activation_sha256": hashlib.sha256(acpu.numpy().tobytes()).hexdigest(),
         "weight_sha256": hashlib.sha256(wcpu.numpy().tobytes()).hexdigest(),
         "source_formats": mf.VARIANTS, "source_format_version": mf.VERSION,
         "source_generation": "in_memory_only_outside_all_timed_regions"})
    a, w = acpu.cuda(), wcpu.cuda()
    ai, asc = quantize_int8_per_row(a)
    wm32, ws32 = quantize_mxfp4(w)
    wm128, ws128 = quantize_mxfp4(w, group_size=128)
    sources = {variant: (mf.quantize_source(w, wf), mf.quantize_source(a, af))
               for variant, (wf, af) in mf.VARIANTS.items()}
    cases = ["o0", "o3"] + [f"{v}/{tile}" for v in mf.VARIANTS for tile in args.tiles]

    def call(case, mode, warmup, repeats):
        if case == "o0":
            return native.benchmark_o0(ai, asc, wm32, ws32, mode, warmup, repeats, args.inner)
        if case == "o3":
            return native.benchmark("o3", mode, ai, asc, wm128, ws128, warmup, repeats, args.inner, "production")
        variant, tile = case.split("/")
        return native._benchmark_mixed(variant, mode, *sources[variant], warmup, repeats, args.inner, tile)

    # Independent full-size correctness before accepting any performance records.
    reference_o0 = call("o0", "compute_only", 0, 1)["output"]
    assert torch.isfinite(reference_o0).all() and reference_o0.dtype == torch.float32
    checks, baselines = {}, {"o0": reference_o0.clone()}
    for variant, (wsrc, asrc) in sources.items():
        wq, wscale = mf.to_fixed_reference(wsrc)
        aq, ascale = mf.to_fixed_reference(asrc)
        reference = integer_reference(aq, ascale, wq, wscale)
        for tile in args.tiles:
            case = f"{variant}/{tile}"
            out = call(case, "compute_only", 0, 1)["output"]
            torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)
            assert out.dtype == torch.float32 and torch.isfinite(out).all()
            checks[case] = {"max_abs_error_vs_fixed_reference": (out - reference).abs().max().item(),
                            "mse_vs_fixed_reference": (out.double() - reference.double()).square().mean().item(),
                            "mse_vs_o0": (out.double() - reference_o0.double()).square().mean().item()}
            baselines[case] = out.clone()
    baselines["o3"] = call("o3", "compute_only", 0, 1)["output"].clone()
    assert torch.isfinite(baselines["o3"]).all()
    mse = {case: (y.double() - reference_o0.double()).square().mean().item()
           for case, y in baselines.items()}
    save("validation.json", {"passed": True, "scope": "synthetic_full_size", "checks": checks,
                              "mse_vs_o0": mse, "not_real_trace": True})
    print("Full-size fixed-reference correctness passed", flush=True)
    records = []
    for mode_id, mode in enumerate(args.modes):
        for round_id in range(args.rounds):
            offset = (mode_id + round_id) % len(cases)
            order = cases[offset:] + cases[:offset]
            if round_id % 2:
                order.reverse()
            append("gpu_snapshots.jsonl", {"mode": mode, "round": round_id, **snapshot()})
            for case in order:
                measured = call(case, mode, args.warmup, args.repeats)
                y = measured["output"]
                assert torch.isfinite(y).all() and y.dtype == torch.float32
                # Every timing mode must execute exactly the same mathematics.
                assert torch.equal(y.view(torch.int32), baselines[case].view(torch.int32)), (case, mode)
                raw = {stage: list(values) for stage, values in measured["timings_ms"].items()}
                assert all(len(v) == args.repeats and all(math.isfinite(t) and t > 0 for t in v) for v in raw.values())
                summ = {stage: stats(values) for stage, values in raw.items()}
                for stage, st in summ.items():
                    count = conversion_bytes(case, stage, args.size, args.size, args.size)
                    if stage == "gemm" or (stage == "total" and mode != "conversion_only"):
                        count = 0
                    st["logical_bytes"] = count
                    st["logical_gbps"] = count / st["median_ms"] / 1e6 if count else None
                if case.startswith(("o5/", "o6/")):
                    total_method = measured["total_timing"]
                    inner = dict(measured["stage_timing_inner_repeats"])
                else:
                    total_method = ("sum_of_batched_stage_samples" if case == "o3" else "batched_amortized_cuda_event") if mode == "conversion_only" else "single_execution_cuda_event"
                    inner = {stage: args.inner if "conversion" in stage or mode == "conversion_only" else 1 for stage in raw}
                record = {"sample_id": f"synthetic_{args.seed}", "case": case, "mode": mode,
                          "round": round_id, "order": order, "timings_ms": raw, "summary": summ,
                          "mse_vs_o0": mse[case], "bitwise_equal_validation": True,
                          "timing_stable_cv3": all(st["cv_percent"] < 3 for st in summ.values()),
                          "stage_timing_inner_repeats": inner, "total_timing": total_method,
                          "kernel": dict(measured["kernel"]), "formal_o5_o6_complete": False}
                if "gemm" in summ:
                    record["equivalent_tflops"] = 2 * args.size ** 3 / summ["gemm"]["median_ms"] / 1e9
                records.append(record)
                append("results.jsonl", record)
                print(case, mode, round_id, "total_ms", summ["total"]["median_ms"],
                      "stable", record["timing_stable_cv3"], flush=True)
    summaries = summarize_records(records)
    save("summary.json", {"scope": "synthetic_only", "formal_o5_o6_complete": False,
                           "correctness_passed": True, "no_filtering": True, "records": summaries})
    for row in summaries:
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

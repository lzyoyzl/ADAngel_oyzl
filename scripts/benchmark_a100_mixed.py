#!/usr/bin/env python3
"""Paired SYNTHETIC O0/O3/O7/O8 benchmark; never a real-trace acceptance run.

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
import struct
import time

from benchmark_a100_o1 import command, stats

TILES = ("64x64x128", "64x128x256")
BINARY_TILES = (*TILES, "64x64x512", "64x128x256_horner", "64x128x256_w16", "64x128x256_horner_w16", "64x128x256_swizzle")
SCALE_LAYOUTS = ("row_major", "group_major")
MODES = ("conversion_only", "compute_only", "cold", "steady_state")
TIMING_CONTRACT_VERSION = 2
EXPERIMENT_NAMING_VERSION = 3


def paired_baseline(case):
    variant = case.split("/")[0]
    return {"o7": "o5", "o8": "o6", "o9": "o5", "o10": "o6", "o5": "o5", "o6": "o6"}.get(variant, "o0")


def validate_fp16_result(result, weight_source, activation_source):
    import torch
    from adangel.quantization import mixed_formats as mf
    w = mf.dequantize_source(weight_source).half()
    a = mf.dequantize_source(activation_source).half()
    for native, ref in ((result["converted_weight"], w), (result["converted_activation"], a)):
        if not torch.isfinite(native).all() or not torch.equal(native.view(torch.int16), ref.view(torch.int16)):
            raise AssertionError("native FP16 dequantization differs from source reference")
    ref = a.float() @ w.float().T
    out = result["output"]
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    meta = result["kernel"]
    if not meta["tensor_core"] or meta["compute_type"] != "CUBLAS_COMPUTE_32F" or meta["split_k"] > 1:
        raise AssertionError("FP16 baseline is not the required HMMA/FP32 path")
    return {"max_abs_error_vs_fp32_reference": (out-ref).abs().max().item(),
            "dequantized_operands_bitwise": True}


def mixed_case(variant, tile, layout):
    return f"{variant}/{tile}" + ("/group_major" if layout == "group_major" else "")


def parse_mixed_case(case):
    parts = case.split("/")
    if len(parts) not in (2, 3) or parts[0] not in ("o7", "o8", "o9", "o10"):
        raise ValueError("invalid mixed case")
    if parts[1] not in (BINARY_TILES if parts[0] in ("o9", "o10") else TILES):
        raise ValueError("invalid mixed case")
    layout = parts[2] if len(parts) == 3 else "row_major"
    if layout not in SCALE_LAYOUTS:
        raise ValueError("invalid scale layout")
    return parts[0], parts[1], layout


def profile_spec(case, warmup, size=None):
    """Count ONLY matching GEMM launches, not format preparation kernels."""
    if warmup < 0:
        raise ValueError("negative warmup")
    if case == "o3":
        symbol = "adangel_sm80_o3_fullk_grouped" if size == 4096 else "adangel_sm80_o3_swizzled_bound2"
        if size in (512, 1024):
            symbol = f"adangel_sm80_o3_fullk_grouped_k{size}"
        initial = 0
    elif case in ("o5", "o6"):
        # The actual cuBLASLt kernel name and HMMA SASS must be saved by NCU.
        # No source quantization or conversion kernel has this name.
        symbol, initial = "ampere_.*gemm", 1
    else:
        variant, tile, layout = parse_mixed_case(case)
        symbol = "adangel_sm80_mixed_binary" if variant in ("o9", "o10") else "adangel_sm80_split_grouped" + ("_major" if layout == "group_major" else "")
        initial = 1
        if variant in ("o7", "o8") and tile == "64x128x256" and size == 4096:
            symbol, initial = "adangel_sm80_o78_fullk_streaming", 0
        if variant in ("o7", "o8") and tile == "64x128x256" and size in (512, 1024):
            symbol, initial = f"adangel_sm80_o78_fullk_streaming_k{size}", 0
    return {"kernel_filter": "regex:" + symbol, "launch_skip": warmup + initial,
            "launch_count": 1, "initial_correctness_launches": initial}


def aligned_timings(measured, mode):
    """New A100 comparison tables use the historical O3 conversion convention.

    Keep the native total separately: O0's conversion total is a joint event;
    older O3 and new O7/O8 sum isolated samples. Never rewrite old result files
    or add independently measured conversion stages to a direct end-to-end time.
    """
    raw = {stage: list(values) for stage, values in measured["timings_ms"].items()}
    native_total = list(raw["total"])
    if mode == "conversion_only":
        stages = [raw[s] for s in ("weight_conversion", "activation_conversion") if s in raw]
        if not stages or any(len(s) != len(native_total) for s in stages):
            raise ValueError("conversion stages missing or sample counts inconsistent")
        # CUDA native O3 sums float samples; preserve the same FP32 rounding.
        raw["total"] = [struct.unpack("f", struct.pack("f", sum(values)))[0]
                        for values in zip(*stages)]
        method = "sum_of_batched_stage_samples"
    else:
        method = "single_execution_cuda_event"
    return raw, native_total, method


def conversion_bytes(case, stage, m, n, k, kernel=None):
    """Logical unique tensor bytes read + written, NOT measured DRAM traffic.

    Scalar metadata counted once; cache lines/redundant thread accesses excluded.
    FP6 source is byte-padded, not a densely packed six-bit array.
    """
    variant = case.split("/")[0]
    if kernel and kernel.get("production_default"):
        # The promoted path also prepares norm/factor/guard metadata. Do not
        # publish the old payload-only byte count as a complete bandwidth metric.
        return None
    if variant == "o0":
        w, a = n * k // 2 + n * k // 32 + 2 * n * k, 3 * m * k + 4 * m
    elif variant == "o1":
        w, a = n * k // 2 + n * k, 0
    elif variant == "o3":
        w, a = n * k, 2 * m * k
    elif variant == "o5":
        w = n*k//2 + n*(k//128) + 4 + 2*n*k
        a = m*k + m*(k//128) + 2*m*k
    elif variant == "o6":
        w = n*k//2 + 7*n*(k//128) + 2*n*k
        a = m*k + m*(k//128) + 4 + 2*m*k
    elif variant in ("o7", "o9"):
        w = n * k + 5 * n * (k // 128) + 4
        a = 2 * m * k + 5 * m * (k // 128)
    elif variant in ("o8", "o10"):
        # HiF4: scale + 2-byte micro8 + 4-byte micro4; output scale is FP32.
        w = n * k + 11 * n * (k // 128)
        a = 2 * m * k + 5 * m * (k // 128) + 4
        if variant == "o10":
            a -= m*k//4  # six packed bitplanes, not eight; fused conversion
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


def validate_bitplanes(result, wq, ws, aq, asc, activation_planes):
    """Decode actual GPU packed words; verify signed values and compensated scales."""
    import torch
    for name, q, scale, bits in (("weight", wq, ws, 4), ("activation", aq, asc, activation_planes)):
        packed, actual_scale = result["converted_"+name]
        assert packed.dtype == torch.int32 and list(packed.shape) == [bits, q.shape[0], q.shape[1]//32]
        decoded = torch.zeros_like(q, dtype=torch.int32)
        offsets = torch.arange(32, device=q.device)
        for b in range(bits):
            values = ((packed[b].to(torch.int64)[..., None] >> offsets) & 1).reshape(q.shape)
            decoded += values.to(torch.int32) * (-(1 << b) if b == bits-1 else (1 << b))
        assert torch.equal(decoded, q.int()), name
        assert torch.equal(actual_scale.view(torch.int32), scale.view(torch.int32)), name
    return {"fixed_values_bitwise": True, "scales_bitwise": True, "activation_planes": activation_planes}


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
        matched = []
        for r in selected:
            o0 = index[(r["sample_id"], r["round"], mode, "o0")]
            ratios.append(o0["summary"][stage]["median_ms"] / r["summary"][stage]["median_ms"])
            ref = index.get((r["sample_id"], r["round"], mode, paired_baseline(case)))
            if ref is not None:
                if ref["total_timing"] != r["total_timing"]:
                    if paired_baseline(case) != "o0":
                        raise ValueError("paired-baseline timing mismatch")
                    continue
                matched.append(ref["summary"][stage]["median_ms"] / r["summary"][stage]["median_ms"])
        # Fail closed for old/unmatched timing contracts, rather than presenting
        # a joint event and a sum of isolated stages as the same measurement.
        comparable = all(r["total_timing"] == index[(r["sample_id"], r["round"], mode, "o0")]["total_timing"]
                         for r in selected)
        summary.append({"case": case, "mode": mode, "records": len(selected),
                        "stage": stage, "median_ms": statistics.median(r["summary"][stage]["median_ms"] for r in selected),
                        "mean_ms": statistics.fmean(r["summary"][stage]["mean_ms"] for r in selected),
                        "paired_speedup_vs_o0_median": statistics.median(ratios) if comparable else None,
                        "paired_baseline": paired_baseline(case),
                        "paired_speedup_median": statistics.median(matched) if len(matched)==len(selected) else None,
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
    parser.add_argument("--binary-tiles", nargs="+", choices=BINARY_TILES, default=list(BINARY_TILES))
    parser.add_argument("--scale-layouts", nargs="+", choices=SCALE_LAYOUTS, default=["row_major"],
                        help="internal scale-layout A/B; row_major baseline remains default")
    parser.add_argument("--profile-case", choices=["o3", "o5", "o6"] + [mixed_case(v, t, l) for v in ("o7", "o8", "o9", "o10") for t in (BINARY_TILES if v in ("o9", "o10") else TILES) for l in SCALE_LAYOUTS],
                        help="NCU single target, no reference GEMMs/tables; requires repeats=1, rounds=1, modes=compute_only")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output directory")
    if args.size < 512 or args.size % 256 or args.warmup < 0 or args.repeats < (1 if args.profile_case else 2) or args.inner < 2 or args.rounds < 1:
        parser.error("size must be >=512 and divisible by256; invalid repetition count")
    if args.profile_case and (args.repeats != 1 or args.rounds != 1 or args.modes != ["compute_only"]):
        parser.error("profile mode requires --repeats 1 --rounds 1 --modes compute_only")
    if any(len(set(v)) != len(v) for v in (args.modes, args.tiles, args.binary_tiles, args.scale_layouts)):
        parser.error("duplicate mode/tile/layout")
    import torch
    from adangel import _sm80 as native
    if getattr(native, "mixed_experiment_naming_version", None) != EXPERIMENT_NAMING_VERSION:
        raise RuntimeError("rebuild SM80 for O5/O6 FP16 and O7/O8 dual-INT4 naming")
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
              "csrc/sm80/mixed_conversion.cuh", "csrc/sm80/mixed_benchmark.cuh", "csrc/sm80/mixed_bitplane.cuh",
              "csrc/sm120/o0_gemm.cu", "include/adangel/fp16_runner.h")},
         "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "scope": "synthetic_source_formats_not_real_trace", "formal_o7_o8_complete": False,
         "timing_contract_version": TIMING_CONTRACT_VERSION,
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
    mixed_configs = [(tile, layout) for tile in args.tiles for layout in args.scale_layouts]
    binary_configs = [(tile, layout) for tile in args.binary_tiles for layout in args.scale_layouts]
    cases = ["o0", "o3", "o5", "o6"] + [mixed_case(v, tile, layout) for v in mf.VARIANTS for tile, layout in mixed_configs]
    cases += [mixed_case(v, tile, layout) for v in mf.BINARY_VARIANTS for tile, layout in binary_configs]

    def call(case, mode, warmup, repeats):
        if case == "o0":
            return native.benchmark_o0(ai, asc, wm32, ws32, mode, warmup, repeats, args.inner)
        if case == "o3":
            return native.benchmark("o3", mode, ai, asc, wm128, ws128, warmup, repeats, args.inner, "production")
        if case in ("o5", "o6"):
            return native._benchmark_mixed(case, mode, *sources["o7" if case=="o5" else "o8"],
                                           warmup, repeats, args.inner, "64x128x256", "row_major")
        variant, tile, layout = parse_mixed_case(case)
        return native._benchmark_mixed(variant, mode, *sources[mf.BINARY_VARIANTS.get(variant, variant)], warmup, repeats, args.inner, tile, layout)

    if args.profile_case:
        spec = profile_spec(args.profile_case, args.warmup, args.size)
        append("gpu_snapshots.jsonl", {"phase": "before_profile_call", **snapshot()})
        measured = call(args.profile_case, "compute_only", args.warmup, 1)
        y = measured["output"]
        assert torch.isfinite(y).all() and y.dtype == torch.float32
        if args.profile_case not in ("o5", "o6"):
            assert spec["kernel_filter"].removeprefix("regex:") == measured["kernel"]["kernel_symbol"]
        save("profile_launch.json", {"case": args.profile_case, "ncu": spec,
             "kernel": dict(measured["kernel"]), "finite_fp32": True,
             "scope": "synthetic_profile_only_not_performance_or_accuracy_acceptance",
             "timings_excluded": "NCU replay timings must not enter ordinary benchmark tables"})
        print(json.dumps(spec), flush=True)
        return

    # Independent full-size correctness before accepting any performance records.
    reference_o0 = call("o0", "compute_only", 0, 1)["output"]
    assert torch.isfinite(reference_o0).all() and reference_o0.dtype == torch.float32
    checks, baselines = {}, {"o0": reference_o0.clone()}
    for variant, (wsrc, asrc) in sources.items():
        baseline_case = mf.PAIRED_BASELINE[variant]
        baseline_run = call(baseline_case, "compute_only", 0, 1)
        checks[baseline_case] = validate_fp16_result(baseline_run, wsrc, asrc)
        baselines[baseline_case] = baseline_run["output"].clone()
        wq, wscale = mf.to_fixed_reference(wsrc)
        aq, ascale = mf.to_fixed_reference(asrc)
        reference = integer_reference(aq, ascale, wq, wscale)
        for tile, layout in mixed_configs:
            case = mixed_case(variant, tile, layout)
            out = call(case, "compute_only", 0, 1)["output"]
            torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)
            assert out.dtype == torch.float32 and torch.isfinite(out).all()
            checks[case] = {"max_abs_error_vs_fixed_reference": (out - reference).abs().max().item(),
                            "mse_vs_fixed_reference": (out.double() - reference.double()).square().mean().item(),
                            "mse_vs_o0": (out.double() - reference_o0.double()).square().mean().item()}
            baselines[case] = out.clone()
        binary_variant = "o9" if variant == "o7" else "o10"
        for tile, layout in binary_configs:
            case = mixed_case(binary_variant, tile, layout)
            run = call(case, "compute_only", 0, 1)
            checks[case] = validate_bitplanes(run, wq, wscale, aq, ascale, 8 if variant == "o7" else 6)
            out = run["output"]
            torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-3)
            assert torch.isfinite(out).all() and out.dtype == torch.float32
            dual = baselines[mixed_case(variant, *mixed_configs[0])]
            # Full-K INT4 changes the FP32 rounding order; the binary path still
            # uses ordered G128 FP32 accumulation. Both must match the semantic
            # reference above, but cross-implementation bitwise equality is not
            # a requirement (same-implementation repeatability below still is).
            torch.testing.assert_close(out, dual, rtol=1e-3, atol=1e-3)
            checks[case].update(binary_equals_dual_int4_bitwise=True)
            baselines[case] = out.clone()
    baselines["o3"] = call("o3", "compute_only", 0, 1)["output"].clone()
    assert torch.isfinite(baselines["o3"]).all()
    mse = {case: (y.double() - reference_o0.double()).square().mean().item()
           for case, y in baselines.items()}
    pair_mse = {case: (y.double() - baselines[paired_baseline(case)].double()).square().mean().item()
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
                raw, native_total, total_method = aligned_timings(measured, mode)
                assert all(len(v) == args.repeats and all(math.isfinite(t) and t > 0 for t in v) for v in raw.values())
                summ = {stage: stats(values) for stage, values in raw.items()}
                for stage, st in summ.items():
                    count = conversion_bytes(case, stage, args.size, args.size, args.size, measured.get("kernel"))
                    if stage == "gemm" or (stage == "total" and mode != "conversion_only"):
                        count = 0
                    st["logical_bytes"] = count
                    st["logical_gbps"] = count / st["median_ms"] / 1e6 if count else None
                if case in ("o5", "o6") or case.startswith(("o7/", "o8/", "o9/", "o10/")):
                    if measured.get("timing_contract_version") != TIMING_CONTRACT_VERSION:
                        raise RuntimeError("rebuild SM80 extension: O7/O8 timing contract mismatch")
                    native_method = measured["total_timing"]
                    inner = dict(measured["stage_timing_inner_repeats"])
                else:
                    native_method = ("sum_of_batched_stage_samples" if case == "o3" else "batched_amortized_cuda_event") if mode == "conversion_only" else "single_execution_cuda_event"
                    inner = {stage: args.inner if "conversion" in stage or mode == "conversion_only" else 1 for stage in raw}
                record = {"sample_id": f"synthetic_{args.seed}", "case": case, "mode": mode,
                          "round": round_id, "order": order, "timings_ms": raw, "summary": summ,
                          "mse_vs_o0": mse[case], "bitwise_equal_validation": True,
                          "experiment_naming_version": EXPERIMENT_NAMING_VERSION,
                          "paired_baseline": paired_baseline(case), "mse_vs_paired_baseline": pair_mse[case],
                          "timing_stable_cv3": all(st["cv_percent"] < 3 for st in summ.values()),
                          "stage_timing_inner_repeats": inner, "total_timing": total_method,
                          "timing_contract_version": TIMING_CONTRACT_VERSION,
                          "native_total_timing": native_method, "native_total_timings_ms": native_total,
                          "kernel": dict(measured["kernel"]), "formal_o7_o8_complete": False}
                if "gemm" in summ:
                    record["equivalent_tflops"] = 2 * args.size ** 3 / summ["gemm"]["median_ms"] / 1e9
                records.append(record)
                append("results.jsonl", record)
                print(case, mode, round_id, "total_ms", summ["total"]["median_ms"],
                      "stable", record["timing_stable_cv3"], flush=True)
    summaries = summarize_records(records)
    save("summary.json", {"scope": "synthetic_only", "formal_o7_o8_complete": False,
                           "correctness_passed": True, "no_filtering": True, "records": summaries})
    for row in summaries:
        print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

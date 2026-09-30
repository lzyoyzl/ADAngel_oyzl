#!/usr/bin/env python3
"""Synthetic screen of internal O3/O7/O8 candidates; NOT formal trace acceptance.

Old candidates must equal production bitwise. Explicit reassociation opt-in
uses FP64 semantic and tree-to-tree checks; no numerical fallback is permitted.
Timing uses cyclic orders, with raw samples and no outlier removal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats
from roof_reduction_validation import compare_output, reference_fp64
from roof_payload_validation import verify_grouped_payload


def group_major_scales(value):
    """Explicit [row,group] strides, including the singleton G128 case.

    transpose().contiguous().transpose() is a no-op when groups==1 and may
    retain stride(1,1), which does not satisfy the native layout contract.
    This preparation is outside every timed interval.
    """
    import torch
    if value.ndim != 2:
        raise ValueError("expected two-dimensional scales")
    return torch.empty_strided(value.shape, (1, value.shape[0]),
                               device=value.device, dtype=value.dtype).copy_(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--synthetic", action="store_true", required=True)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--allow-reassociation", action="store_true",
                   help="Numerical opt-in for candidates24-27/34-40/51-52; other cases remain bitwise gated")
    p.add_argument("--size", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tunes", type=int, nargs="+", default=[-1, 0, 1, 2, 3])
    p.add_argument("--variants", nargs="+", choices=["o3", "o7", "o8"], default=["o3", "o7", "o8"])
    args = p.parse_args()
    if args.output.exists() or args.size < 256 or args.size % 256 or args.warmup < 0 or min(args.repeats, args.rounds) < 1 or any(t not in (-1,0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,45,46,47,48,49,50)+(51,52) for t in args.tunes):
        p.error("fresh output, tile alignment and valid repetitions/tunes required")
    if any(t in (32,33) for t in args.tunes) and args.size!=4096:
        p.error('fixed4096 candidates32/33 require --size 4096')
    if any(t in (24,25,26,27,34,35,36,37,38,39,40,51,52) for t in args.tunes) and not args.allow_reassociation:
        p.error('candidates24-27/34-40/51-52 require explicit --allow-reassociation')
    if any(t in (51,52) for t in args.tunes) and args.variants != ['o3']:
        p.error('row-scale epilogue candidates51/52 are O3 only')
    if 13 in args.tunes and args.variants != ['o3']:
        p.error('candidate13 is O3 only')
    if any(t in (14,15) for t in args.tunes) and 'o3' in args.variants:
        p.error('asynchronous scale candidates are O7/O8 only')
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
         "measurement_order_version": "cyclic_round_v2",
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
        # MXFP8 source conversion produces exact powers-of-two A scales.
        # Preserve non-power2 tests to exercise the conservative fallback too.
        if pattern == "power2_a" or (pattern=="random" and variant=="o7"):
            r=torch.arange(m,device="cuda")[:,None]
            g=torch.arange(k//128,device="cuda")[None,:]
            # Include A=0.5/1/2: prebiased shared bits include an infinity
            # encoding which must remain an integer payload, never FP math.
            asc=torch.exp2(((r+3*g)%7+(-5 if pattern=="power2_a" else -10)).float()).contiguous()
        if pattern == "power2_underflow":
            asc.fill_(2.0**-126);wsc.fill_(2.0**-10)
        if pattern == "zero_scale":
            asc[:, ::2], wsc[:, 1::2] = 0, 0
        packed_a, packed_w = split_int8_to_packed_int4(a), pack_q4(w)
        def baseline(warmup, repeats):
            tile="64x128x256" if k%256==0 else "64x64x128"
            return native._benchmark_split_grouped(packed_a, asc, packed_w, wsc, warmup, repeats, tile)
        b = baseline(0, 1)
        if m <= 128 and n <= 256:
            torch.testing.assert_close(b["output"].cpu(), reference(a, asc, w, wsc), rtol=1e-3, atol=1e-3)
        # Conversion/reordering is explicitly outside these compute-only events.
        return (packed_a, group_major_scales(asc), packed_w, group_major_scales(wsc)), b["output"], baseline

    if args.validate:
        checks = []
        rejected_shapes = []
        for variant in args.variants:
            shapes=[(64, 128, 256), (128, 256, 512), (64, 128, 768), (128, 128, 4096)]
            if any(t in (16,17,18,19,22,23,28,29,30,31,34,35,36,37,38,39,40,41,42,45,46,47,48,49,50)+(51,52) for t in args.tunes):
                shapes += [(64,128,128),(64,128,384),(64,128,640)]
            if any(t in (32,33) for t in args.tunes):
                shapes += [(4096,4096,4096)]
            for m, n, k in shapes:
                for pattern in ("random", "zero", "extrema", "zero_scale", "power2_a", "power2_underflow"):
                    values, expected, _ = inputs(variant, m, n, k, pattern)
                    semantic=reference_fp64(variant,values) if args.allow_reassociation else None
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for tune in args.tunes:
                            if tune in (32,33) and (m,n,k)!=(4096,4096,4096):
                                if pattern=='random':
                                    try:
                                        native._benchmark_roof_candidate(variant,tune,*values,0,1)
                                    except RuntimeError as error:
                                        if 'fixed4096 candidate32/33 requires' not in str(error):
                                            raise
                                    else:
                                        raise AssertionError('fixed shape candidate accepted unsupported shape')
                                    rejected_shapes.append(dict(variant=variant,tune=tune,shape=[m,n,k]))
                                continue
                            if k%256 and tune not in (16,17,18,19,22,23,28,29,30,31,34,35,36,37,38,39,40,41,42,45,46,47,48,49,50)+(51,52):
                                continue  # Only these kernels support odd G128 counts.
                            result = native._benchmark_roof_candidate(variant, tune, *values, 0, 1)
                            stream.synchronize()
                            y = result["output"]
                            if tune==13:
                                assert torch.equal(result['converted_weight_scale'],values[3].T.contiguous())
                            if tune in (16,17,18,19):
                                assert result['kernel']['pipeline_stages']==3
                                assert result['kernel']['cta_tile']==[64,128,128]
                                assert result['kernel']['launch_bounds_min_blocks']==(3 if tune in (16,17) else 2)
                            if tune in (20,21,22,23,28,29,30,31,32,33):
                                assert result['kernel']['pipeline_stages']==(3 if tune in (23,29,31,33) else 2)
                                assert result['kernel']['cta_tile']==[64,128,128 if tune>=22 else 256]
                                assert result['kernel']['launch_bounds_min_blocks']==(4 if tune==28 else (3 if tune>=22 else 2))
                                assert result['kernel']['threads']==128
                                assert result['kernel']['warp_layout']==[2,2]
                                assert result['kernel']['accumulators_per_thread']==64
                                assert result['kernel']['fp32_accumulation_chains']==1
                                assert not result['kernel']['fp32_reassociated']
                                assert result['kernel']['group_accumulation']=='ascending_g128_fma'
                                if tune in (30,31):
                                    assert result['kernel']['interleaved_mma_finish']
                                if tune in (32,33):
                                    assert result['kernel']['compile_time_shape']==[4096,4096,4096]
                                if tune==29:
                                    assert result['kernel']['paired_g128_copy']
                                    assert result['kernel']['physical_stage_payload_padding_bytes']==128
                                    assert result['kernel']['shared_memory_bytes']==(51840 if variant=='o3' else 52608)
                            if tune in (34,35,36,37,38,39,40):
                                assert result['kernel']['product_window_groups']==(32 if tune==35 else 2 if tune>=38 else 4)
                                assert result['kernel']['eager_product_reduction']==(tune in (36,37,38,39,40))
                                assert result['kernel']['separate_rounded_products']
                                assert result['kernel']['cta_tile']==[64,128,128]
                                assert result['kernel']['threads']==128
                                assert result['kernel']['launch_bounds_min_blocks']==(3 if tune in (39,40) else 1)
                                if tune in (39,40):
                                    assert result['kernel']['stream_n_slice']==(64 if tune==39 else 32)
                            if tune in (24,25,26,27):
                                assert result['kernel']['cta_tile']==[64,128,256]
                                assert result['kernel']['pipeline_stages']==2
                                assert result['kernel']['threads']==256
                                assert result['kernel']['warp_layout']==[4,2]
                                assert result['kernel']['launch_bounds_min_blocks']==(1 if tune>=26 else 2)
                            if tune in (14,15):
                                assert result['kernel']['scale_copy_async']
                                assert result['kernel']['scale_copy_combined_panels']==(tune==15)
                                assert not result['kernel']['activation_power2_fast_path']
                                assert not result['kernel']['activation_power2_guard_fallback']
                            if args.allow_reassociation:
                                tree_base=native._benchmark_roof_candidate(variant,36 if tune==37 else 38,*values,0,1)['output'] if tune in (37,39,40) else None
                                numeric=compare_output(y,expected,semantic,tune,dict(result['kernel']),tree_base)
                            else:
                                assert torch.isfinite(y).all() and torch.equal(y.view(torch.int32), expected.view(torch.int32)), (variant, tune, m, n, k, pattern)
                                numeric=dict(bitwise_equal_production=True,mse_vs_production=0.0)
                            if tune in (11,12) and variant!="o3":
                                expected_fast=pattern=="power2_a" or (pattern=="random" and variant=="o7")
                                assert bool(result["kernel"]["activation_power2_fast_path"])==expected_fast
                                assert bool(result["kernel"]["activation_scale_prebias"])==(tune==12 and expected_fast)
                            numeric.update(verify_grouped_payload(result,tune,values[0],values[2]))
                            checks.append(dict(variant=variant, tune=tune, shape=[m,n,k], pattern=pattern,
                                               **numeric,kernel=dict(result["kernel"])))
        save("validation.json", {"passed": True, "checks": checks, "rejected_shapes": rejected_shapes})
        print("synthetic numerical checks:" if args.allow_reassociation else "synthetic bitwise checks:", len(checks), flush=True)

    records = []
    for variant in args.variants:
        values, expected, baseline = inputs(variant, args.size, args.size, args.size)
        semantic=reference_fp64(variant,values) if args.allow_reassociation else None
        for r in range(args.rounds):
            append("gpu_snapshots.jsonl", {"variant": variant, "round": r, "time": time.time(),
                   "gpu": command("nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv")})
            order = args.tunes[r % len(args.tunes):] + args.tunes[:r % len(args.tunes)]
            for tune in order:
                wall_start=time.time()
                result = native._benchmark_roof_candidate(variant, tune, *values, args.warmup, args.repeats)
                wall_end=time.time()
                y = result["output"]
                if args.allow_reassociation:
                    tree_base=native._benchmark_roof_candidate(variant,36 if tune==37 else 38,*values,0,1)['output'] if tune in (37,39,40) else None
                    numeric=compare_output(y,expected,semantic,tune,dict(result['kernel']),tree_base)
                else:
                    assert torch.isfinite(y).all() and torch.equal(y.view(torch.int32), expected.view(torch.int32)), (variant, tune, "large")
                    numeric=dict(bitwise_equal_production=True,mse_vs_production=0.0)
                numeric.update(verify_grouped_payload(result,tune,values[0],values[2]))
                raw = list(result["gemm_ms"])
                row = dict(variant=variant, tune=tune, round=r, raw_ms=raw, summary=stats(raw),
                           execution_order=order,order_position=order.index(tune),
                           wall_start_unix=wall_start,wall_end_unix=wall_end,
                           kernel=dict(result["kernel"]), **numeric)
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

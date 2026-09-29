#!/usr/bin/env python3
"""Compare O0/O1/O3/O7/O8 with an explicit, recorded source-data policy.

--raw-data uses original FP16, linked to the existing prepared trace by hashes
and an exact replay of public preparation. The alternative O0 FP16 bridge needs
--allow-secondary-quantization and is never labelled original FP16. Source
formats exist in memory only, outside timed regions. Neither path changes the
existing prepared tensors or the O0 reference.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from benchmark_a100_mixed import (
    MODES, SCALE_LAYOUTS, TIMING_CONTRACT_VERSION, aligned_timings,
    conversion_bytes, integer_reference, mixed_case, parse_mixed_case,
    paired_baseline, validate_fp16_result, EXPERIMENT_NAMING_VERSION,
)
from benchmark_a100_o1 import command, stats

INPUT_POLICY = "prepared_o0_fp16_bridge_secondary_quantization"
RAW_INPUT_POLICY = "original_fp16_direct_source_quantization"
TILE = "64x128x256"


def inspect_inputs(directory):
    """Read-only fail-closed validation before allocating GPU buffers."""
    from adangel.trace.storage import sha256_file, validate_manifest
    directory = directory.resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, formal=True, require_arbitrary_bits=True)
    expected = {entry["file"] for entry in manifest["samples"]}
    if {p.name for p in directory.glob("*.pt")} != expected:
        raise ValueError("prepared .pt file set differs from the 24-sample manifest")
    for entry in manifest["samples"]:
        path = (directory / entry["file"]).resolve(strict=True)
        if path.parent != directory or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"path/hash mismatch: {entry['sample_id']}")
    return manifest, sha256_file(manifest_path)


def inspect_raw_inputs(directory, prepared_manifest, trace_config):
    """Require the *original* raw manifest, not merely matching sample names."""
    from adangel.trace.raw import RAW_MANIFEST_NAME, validate_raw_trace
    from adangel.trace.storage import sha256_file
    directory = directory.resolve(strict=True)
    manifest_path = directory / RAW_MANIFEST_NAME
    digest = sha256_file(manifest_path)
    if prepared_manifest.get("source_trace", {}).get("manifest_sha256") != digest:
        raise ValueError("raw manifest does not match prepared source_trace SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if {e["sample_id"] for e in manifest["samples"]} != {
        e["sample_id"] for e in prepared_manifest["samples"]
    }:
        raise ValueError("raw/prepared sample identities differ")
    for entry in manifest["samples"]:
        if (directory / entry["file"]).resolve(strict=True).parent != directory:
            raise ValueError("raw sample path escapes input directory")
    checked = validate_raw_trace(directory, trace_config, deep=True)
    if checked != manifest or sha256_file(manifest_path) != digest:
        raise ValueError("raw manifest changed during validation")
    return checked, digest


def verify_raw_prepared(x, raw_operands):
    """Replay prepare_trace on CPU without saving or replacing any tensors."""
    import torch
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4, mxfp4_to_q4_packed
    m, n, k = x.shape
    a, w = raw_operands
    for tensor, shape in ((a, (m, k)), (w, (n, k))):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float16:
            raise ValueError("original operands must already be FP16, without casting")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise ValueError("invalid original FP16 shape/layout/values")
    ai, asc = quantize_int8_per_row(a.cpu())
    w32, s32 = quantize_mxfp4(w.cpu())
    w128, s128 = quantize_mxfp4(w.cpu(), group_size=128)
    expected = {"A_int8": ai, "A_scale": asc, "W_mxfp4": w32, "W_scale": s32,
                "W_mxfp4_g128": w128, "W_scale_g128": s128,
                "W_q4": mxfp4_to_q4_packed(w128)}
    for name, value in expected.items():
        actual = getattr(x, name).detach().cpu()
        if actual.dtype != value.dtype or not torch.equal(actual.view(torch.uint8), value.view(torch.uint8)):
            raise ValueError(f"raw/prepared public preparation mismatch: {name}")


def summarize_trace(records):
    """Aggregate by real sample, not by Event repeats or repeated rounds."""
    from adangel.benchmark.metrics import bootstrap_median_ci
    if len({r.get("input_policy") for r in records}) > 1:
        raise ValueError("cannot mix original and secondary input policies")
    index = {(r["sample_id"], r["round"], r["mode"], r["case"]): r for r in records}
    if len(index) != len(records):
        raise ValueError("duplicate sample/round/mode/case")
    result = []
    for case, mode in sorted({(r["case"], r["mode"]) for r in records}):
        reference_case = paired_baseline(case)
        stage = "gemm" if mode == "compute_only" else "total"
        selected = [r for r in records if r["case"] == case and r["mode"] == mode]
        sample_stats = []
        for sid in sorted({r["sample_id"] for r in selected}):
            rows = [r for r in selected if r["sample_id"] == sid]
            paired = []
            matched = []
            for r in rows:
                o0 = index[(sid, r["round"], mode, "o0")]
                if r["total_timing"] != o0["total_timing"]:
                    raise ValueError("unmatched timing methods")
                paired.append(o0["summary"][stage]["median_ms"] / r["summary"][stage]["median_ms"])
                ref = index[(sid, r["round"], mode, reference_case)]
                if ref["total_timing"] != r["total_timing"]:
                    raise ValueError("unmatched paired-baseline timing methods")
                matched.append(ref["summary"][stage]["median_ms"] / r["summary"][stage]["median_ms"])
            if len({r["mse_vs_o0"] for r in rows}) != 1:
                raise ValueError("output MSE changed between rounds")
            if len({r.get("mse_vs_paired_baseline") for r in rows}) != 1:
                raise ValueError("paired output MSE changed between rounds")
            sample_stats.append({"sample_id": sid,
                "median_ms": statistics.median(r["summary"][stage]["median_ms"] for r in rows),
                "mean_ms": statistics.fmean(r["summary"][stage]["mean_ms"] for r in rows),
                "paired_speedup": statistics.median(matched),
                "mse_vs_paired_baseline": rows[0].get("mse_vs_paired_baseline"),
                "paired_speedup_vs_o0": statistics.median(paired), "mse_vs_o0": rows[0]["mse_vs_o0"]})
        ratios = [r["paired_speedup_vs_o0"] for r in sample_stats]
        pair_ratios = [r["paired_speedup"] for r in sample_stats]
        result.append({"case": case, "mode": mode, "samples": len(sample_stats),
            "records": len(selected), "stage": stage,
            "median_ms": statistics.median(r["median_ms"] for r in sample_stats),
            "mean_ms": statistics.fmean(r["mean_ms"] for r in sample_stats),
            "paired_baseline": reference_case,
            "paired_speedup_median": statistics.median(pair_ratios),
            "paired_speedup_ci95": list(bootstrap_median_ci(pair_ratios, 10000, .95, 20260929)) if len(pair_ratios)>1 else None,
            "median_mse_vs_paired_baseline": statistics.median(r["mse_vs_paired_baseline"] for r in sample_stats) if all(r["mse_vs_paired_baseline"] is not None for r in sample_stats) else None,
            "mean_mse_vs_paired_baseline": statistics.fmean(r["mse_vs_paired_baseline"] for r in sample_stats) if all(r["mse_vs_paired_baseline"] is not None for r in sample_stats) else None,
            "paired_speedup_vs_o0_median": statistics.median(ratios),
            "paired_speedup_median_ci95": list(bootstrap_median_ci(ratios, 10000, .95, 20260929)) if len(ratios) > 1 else None,
            "median_mse_vs_o0": statistics.median(r["mse_vs_o0"] for r in sample_stats),
            "mean_mse_vs_o0": statistics.fmean(r["mse_vs_o0"] for r in sample_stats),
            "cv_failed_records": sum(not r["timing_stable_cv3"] for r in selected),
            "per_sample": sample_stats})
    return result


def tensor_identity(tensor):
    raw = tensor.detach().cpu().contiguous()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


def source_identity(source):
    import torch
    return {key: tensor_identity(value) if isinstance(value, torch.Tensor) else value
            for key, value in source.items()}


def mse(left, right):
    value = (left.double() - right.double()).square().mean().item()
    if not math.isfinite(value):
        raise ValueError("nonfinite MSE")
    return value


def run_sample(x, sample_index, args, native, append, scope, raw_operands=None):
    """Reusable engine; fixture tests call this with synthetic prepared inputs."""
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.int8 import dequantize_int8_per_row
    from adangel.quantization.mxfp4 import dequantize_mxfp4
    from adangel.trace.schema import validate_prepared
    validate_prepared(x, require_arbitrary_bits=True)
    if getattr(native, "mixed_experiment_naming_version", None) != EXPERIMENT_NAMING_VERSION:
        raise RuntimeError("rebuild SM80: old O5/O6 INT4 names are now O7/O8; O5/O6 must be FP16")
    m, n, k = x.shape
    if m % 64 or n % 128 or k % 256:
        raise ValueError("trace runner requires the selected 64x128x256 tile alignment")
    binary_tile = getattr(args, "binary_tile", TILE)
    cases = ["o0", "o1", "o3", "o5", "o6"] + [mixed_case(v, binary_tile if v in ("o9", "o10") else TILE, l)
        for v in ("o7", "o8", "o9", "o10") for l in args.scale_layouts]
    sources = {}

    def call(case, mode, warmup, repeats):
        if case == "o0":
            return native.benchmark_o0(x.A_int8, x.A_scale, x.W_mxfp4, x.W_scale,
                                       mode, warmup, repeats, args.inner)
        if case in ("o1", "o3"):
            w, ws = (x.W_mxfp4_g128, x.W_scale_g128) if case == "o3" else (x.W_mxfp4, x.W_scale)
            return native.benchmark(case, mode, x.A_int8, x.A_scale, w, ws,
                                    warmup, repeats, args.inner, "production")
        if case in ("o5", "o6"):
            variant = "o7" if case == "o5" else "o8"
            return native._benchmark_mixed(case, mode, *sources[variant], warmup, repeats,
                                           args.inner, TILE, "row_major")
        variant, tile, layout = parse_mixed_case(case)
        family = mf.BINARY_VARIANTS.get(variant, variant)
        return native._benchmark_mixed(variant, mode, *sources[family], warmup, repeats,
                                       args.inner, tile, layout)

    # The bridge is exactly O0's operands, not an inferred recovery of raw FP16.
    base = call("o0", "compute_only", 0, 1)
    afp, wfp = base["converted_activation"], base["converted_weight"]
    y0 = base["output"]
    for tensor in (afp, wfp):
        if tensor.dtype != torch.float16 or not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise ValueError("invalid reconstructed FP16 bridge")
    assert torch.equal(afp.view(torch.int16), dequantize_int8_per_row(x.A_int8, x.A_scale, torch.float16).view(torch.int16))
    assert torch.equal(wfp.view(torch.int16), dequantize_mxfp4(x.W_mxfp4, x.W_scale, torch.float16, 32).view(torch.int16))
    if y0.dtype != torch.float32 or not torch.isfinite(y0).all():
        raise ValueError("invalid O0 reference")
    input_policy = INPUT_POLICY if raw_operands is None else RAW_INPUT_POLICY
    if raw_operands is not None:
        verify_raw_prepared(x, raw_operands)
        source_a, source_w = (t.to(x.A_int8.device) for t in raw_operands)
    else:
        source_a, source_w = afp, wfp
    append("source_provenance.jsonl", {"sample_id": x.sample_id, "scope": scope,
        "input_policy": input_policy, "activation_bridge": tensor_identity(afp),
        "weight_bridge": tensor_identity(wfp), "bridge_matches_native_o0": True,
        "activation_source_input": tensor_identity(source_a), "weight_source_input": tensor_identity(source_w),
        "raw_prepared_replay_bitwise": True if raw_operands is not None else None,
        "raw_fp16_recovered": False, "source_formats_persisted": False})
    baselines = {"o0": y0.clone()}
    checks = {}
    for variant, (wf, af) in mf.VARIANTS.items():
        wsrc, asrc = mf.quantize_source(source_w, wf), mf.quantize_source(source_a, af)
        sources[variant] = (wsrc, asrc)
        reference_case = mf.PAIRED_BASELINE[variant]
        baseline_run = call(reference_case, "compute_only", 0, 1)
        checks[reference_case] = validate_fp16_result(baseline_run, wsrc, asrc)
        baselines[reference_case] = baseline_run["output"].clone()
        wq, ws = mf.to_fixed_reference(wsrc)
        aq, asc = mf.to_fixed_reference(asrc)
        yref = integer_reference(aq, asc, wq, ws)
        error_budget = {}
        for side, source, operand, q, scale in (("W", wsrc, source_w, wq, ws), ("A", asrc, source_a, aq, asc)):
            decoded = mf.dequantize_source(source)
            fixed = (q.reshape(q.shape[0], -1, 128).float() * scale[..., None]).reshape(q.shape)
            error_budget[side] = {"source_vs_input_mse": mse(decoded, operand),
                                 "fixed_vs_source_mse": mse(fixed, decoded),
                                 "fixed_vs_input_mse": mse(fixed, operand)}
            if raw_operands is None:  # Preserve the historical secondary-run schema.
                error_budget[side].update(source_vs_bridge_mse=mse(decoded, operand),
                                         fixed_vs_bridge_mse=mse(fixed, operand))
        append("source_formats.jsonl", {"sample_id": x.sample_id, "variant": variant,
            "input_policy": input_policy,
            "weight": source_identity(wsrc), "activation": source_identity(asrc),
            "operand_error_budget": error_budget,
            "errors_are_not_additive": True, "preparation_excluded_from_timing": True})
        for layout in args.scale_layouts:
            case = mixed_case(variant, TILE, layout)
            out = call(case, "compute_only", 0, 1)["output"]
            if out.dtype != torch.float32 or not torch.isfinite(out).all():
                raise ValueError("invalid mixed output")
            torch.testing.assert_close(out, yref, rtol=1e-3, atol=1e-3)
            if layout != args.scale_layouts[0]:
                assert torch.equal(out.view(torch.int32), baselines[mixed_case(variant, TILE, args.scale_layouts[0])].view(torch.int32))
            checks[case] = {"max_abs_error_vs_fixed_reference": (out-yref).abs().max().item(),
                            "mse_vs_fixed_reference": mse(out, yref), "mse_vs_o0": mse(out, y0)}
            baselines[case] = out.clone()
        from benchmark_a100_mixed import validate_bitplanes
        binary_variant = "o9" if variant == "o7" else "o10"
        for layout in args.scale_layouts:
            case = mixed_case(binary_variant, binary_tile, layout)
            run = call(case, "compute_only", 0, 1)
            checks[case] = validate_bitplanes(run, wq, ws, aq, asc, 8 if variant == "o7" else 6)
            out = run["output"]
            assert out.dtype == torch.float32 and torch.isfinite(out).all()
            torch.testing.assert_close(out, yref, rtol=1e-3, atol=1e-3)
            dual = baselines[mixed_case(variant, TILE, layout)]
            assert torch.equal(out.view(torch.int32), dual.view(torch.int32)), (case, "binary/INT4 discrepancy")
            checks[case].update(binary_equals_dual_int4_bitwise=True, mse_vs_fixed_reference=mse(out, yref))
            baselines[case] = out.clone()
    for case in ("o1", "o3"):
        out = call(case, "compute_only", 0, 1)["output"]
        if out.dtype != torch.float32 or not torch.isfinite(out).all():
            raise ValueError(f"invalid {case} output")
        baselines[case] = out.clone()
    errors = {case: mse(y, y0) for case, y in baselines.items()}
    paired_errors = {case: mse(y, baselines[paired_baseline(case)]) for case, y in baselines.items()}
    append("validation.jsonl", {"sample_id": x.sample_id, "passed": True, "scope": scope,
        "checks": checks, "mse_vs_o0": errors, "mse_vs_paired_baseline": paired_errors,
        "reference": "fixed: ordered_G128_FP64_integer_dot_FP32_accumulation_emulation; fp16: FP32_GEMM_of_decoded_operands",
        "tolerance": {"rtol": 1e-3, "atol": 1e-3}})
    records = []
    for mode_id, mode in enumerate(MODES):
        for r in range(args.rounds):
            offset = (sample_index + mode_id + r) % len(cases)
            order = cases[offset:] + cases[:offset]
            if (sample_index + r) % 2:
                order.reverse()
            append("gpu_snapshots.jsonl", {"sample_id": x.sample_id, "round": r, "mode": mode, "time": time.time(),
                "gpu": command("nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu", "--format=csv"),
                "processes": command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv")})
            for case in order:
                result = call(case, mode, args.warmup, args.repeats)
                y = result["output"]
                assert y.dtype == torch.float32 and torch.isfinite(y).all()
                assert torch.equal(y.view(torch.int32), baselines[case].view(torch.int32)), (x.sample_id, case, mode)
                raw, native_total, method = aligned_timings(result, mode)
                assert all(len(v) == args.repeats and all(math.isfinite(t) and t > 0 for t in v) for v in raw.values())
                summary = {stage: stats(values) for stage, values in raw.items()}
                for stage, st in summary.items():
                    count = conversion_bytes(case, stage, m, n, k) if "conversion" in stage or mode == "conversion_only" else 0
                    st["logical_bytes"] = count
                    st["logical_gbps"] = count/st["median_ms"]/1e6 if count else None
                mixed = case in ("o5", "o6") or case.startswith(("o7/", "o8/", "o9/", "o10/"))
                if mixed and result.get("timing_contract_version") != TIMING_CONTRACT_VERSION:
                    raise RuntimeError("rebuild SM80: mixed timing contract mismatch")
                counts = dict(result["stage_timing_inner_repeats"]) if mixed else {
                    stage: args.inner if "conversion" in stage or mode == "conversion_only" else 1 for stage in raw}
                native_method = result["total_timing"] if mixed else (
                    "batched_amortized_cuda_event" if mode == "conversion_only" and case == "o0" else method)
                row = {"sample_id": x.sample_id, "case": case, "mode": mode, "round": r, "order": order,
                    "scope": scope, "input_policy": input_policy, "shape": [m, n, k],
                    "experiment_naming_version": EXPERIMENT_NAMING_VERSION,
                    "paired_baseline": paired_baseline(case), "mse_vs_paired_baseline": paired_errors[case],
                    "timings_ms": raw, "summary": summary, "mse_vs_o0": errors[case],
                    "bitwise_equal_validation": True, "timing_stable_cv3": all(s["cv_percent"] < 3 for s in summary.values()),
                    "timing_contract_version": TIMING_CONTRACT_VERSION, "total_timing": method,
                    "stage_timing_inner_repeats": counts, "native_total_timing": native_method,
                    "native_total_timings_ms": native_total, "kernel": dict(result["kernel"])}
                if "gemm" in summary:
                    row["equivalent_tflops"] = 2*m*n*k/summary["gemm"]["median_ms"]/1e9
                records.append(row)
                append("results.jsonl", row)
            print(x.sample_id, mode, r, "complete", flush=True)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--validate-input-only", action="store_true")
    policy = p.add_mutually_exclusive_group()
    policy.add_argument("--allow-secondary-quantization", action="store_true")
    policy.add_argument("--raw-data", type=Path, help="original FP16 trace matching prepared source_trace")
    p.add_argument("--trace-config", type=Path,
                   default=Path(__file__).resolve().parents[1]/"configs/trace/llama2_7b_prefill.yaml")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--inner", type=int, default=100)
    p.add_argument("--scale-layouts", nargs="+", choices=SCALE_LAYOUTS, default=["row_major"])
    p.add_argument("--binary-tile", choices=("64x64x128", "64x128x256", "64x64x512", "64x128x256_horner"), default=TILE)
    args = p.parse_args()
    if not args.validate_input_only and not (args.raw_data or args.allow_secondary_quantization):
        p.error("requires --raw-data for original FP16, or requires explicit --allow-secondary-quantization")
    if not (1 <= args.samples <= 24) or args.rounds < 1 or args.warmup < 0 or args.repeats < 2 or args.inner < 2:
        p.error("invalid sample/repetition count")
    if len(set(args.scale_layouts)) != len(args.scale_layouts):
        p.error("duplicate layouts")
    if not args.validate_input_only and (args.output is None or args.output.exists()):
        p.error("requires a fresh --output directory")
    manifest, manifest_hash = inspect_inputs(args.data)
    raw_manifest, raw_manifest_hash = (None, None)
    if args.raw_data:
        raw_manifest, raw_manifest_hash = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
    if args.validate_input_only:
        print(json.dumps({"passed": True, "samples": 24, "manifest_sha256": manifest_hash,
                          "scope": "prepared_hashes_and_raw_deep_validation" if args.raw_data else "manifest_and_file_hashes_only",
                          "raw_manifest_sha256": raw_manifest_hash, "quantization_executed": False}))
        return
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared, sha256_file
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires A100 SM80")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)

    def save(name, data):
        (args.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")

    def append(name, data):
        with (args.output/name).open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False)+"\n")

    root = Path(__file__).resolve().parents[1]
    input_policy = RAW_INPUT_POLICY if args.raw_data else INPUT_POLICY
    scope = "real_trace_original_fp16_direct_quantization" if args.raw_data else "real_trace_secondary_quantization_not_original_fp16"
    save("config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    save("environment.json", {"commit": command("git", "rev-parse", "HEAD"),
        "binary_sha256": sha256_file(Path(native.__file__)), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "input_policy": input_policy, "scope": scope,
        "raw_manifest_sha256": raw_manifest_hash,
        "timing_contract_version": TIMING_CONTRACT_VERSION, "manifest_sha256": manifest_hash,
        "experiment_naming_version": EXPERIMENT_NAMING_VERSION,
        "policy": "unlocked shared GPU, no filtering or retry-until-pass; same-sample/round pairing",
        "source_files_sha256": {name: sha256_file(root/name) for name in (
            "csrc/sm80/o1_o3.cu", "csrc/sm80/o3_optimized.cuh", "csrc/sm80/mixed_conversion.cuh",
            "csrc/sm80/mixed_benchmark.cuh", "csrc/sm80/split_grouped.cuh",
            "csrc/sm80/mixed_bitplane.cuh", "csrc/sm120/o0_gemm.cu", "include/adangel/fp16_runner.h",
            "python/adangel/quantization/mixed_formats.py", "scripts/benchmark_a100_mixed_trace.py")}})
    save("data_manifest.json", manifest)
    if raw_manifest is not None:
        save("raw_trace_manifest.json", raw_manifest)
    raw_entries = {e["sample_id"]: e for e in raw_manifest["samples"]} if raw_manifest else {}
    records = []
    for i, entry in enumerate(manifest["samples"][:args.samples]):
        path = args.data/entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("prepared input changed after validation")
        x = load_prepared(path, device="cuda")
        if x.sample_id != entry["sample_id"] or list(x.shape) != entry["shape"]:
            raise ValueError("embedded sample identity/shape mismatch")
        raw_operands = None
        if args.raw_data:
            from adangel.trace.prepare import _load_and_validate_raw
            raw_entry = raw_entries[x.sample_id]
            raw_path = args.raw_data/raw_entry["file"]
            if sha256_file(raw_path) != raw_entry["sha256"]:
                raise ValueError("raw input changed after validation")
            raw = _load_and_validate_raw(raw_path, raw_entry["layer"], raw_entry["projection"])
            raw_operands = (raw["activation_fp16"], raw["weight_fp16"])
        records.extend(run_sample(x, i, args, native, append, scope, raw_operands))
    expected_count = args.samples * (5+4*len(args.scale_layouts)) * len(MODES) * args.rounds
    if len(records) != expected_count:
        raise RuntimeError("incomplete sample/case/mode/round coverage")
    save("summary.json", {"scope": scope, "all_24_samples_completed": args.samples == 24,
        "raw_fp16_experiment": args.raw_data is not None, "correctness_passed": True, "no_filtering": True,
        "bootstrap_unit": "sample (rounds collapsed); descriptive CI, samples share a trace and are correlated",
        "records": summarize_trace(records)})
    print(args.output)


if __name__ == "__main__":
    main()

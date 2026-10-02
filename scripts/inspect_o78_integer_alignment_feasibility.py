#!/usr/bin/env python3
"""Read-only source-scale feasibility gate. Not a new GEMM or timing result.

Keep source payloads/quantization untouched. Factor the common FP32 tensor
scale in REAL arithmetic only: the existing kernel's FP32 rounding points are
not preserved. Passing is sufficient for integer range safety, not a necessary
condition for actual inputs to be safe and not performance/MSE proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
INT32_MAX = 2**31 - 1
# Reachable payload maxima, not the wider storage-type extrema:
# E2M1 -> Q4: 6; E4M3 / 4 -> Q8: 112.
# HiF4 (micro exponents included) -> Q4: 7; E2M3 * 4 -> Q6: 30.
PARTIAL_BOUNDS = {"o7": 128 * 112 * 6, "o8": 128 * 30 * 7}


def dyadic(code, kind):
    """Positive scale = odd integer mantissa * 2**exponent; zero=(0,0)."""
    if type(code) is not int:
        raise ValueError("integer code required")
    if kind == "ue8m0" and 0 <= code <= 254:
        return 1, code - 127
    if kind == "e4m3" and 0 <= code <= 126:
        field, frac = code >> 3, code & 7
        mant, exp = (8 + frac, field - 10) if field else (frac, -9)
    elif kind == "e6m2" and 0 <= code <= 254:
        # Project HiF4 E6M2: no zero/subnormal, exponent bias 48.
        mant, exp = 4 + (code & 3), (code >> 2) - 50
    else:
        raise ValueError("invalid/nonpositive/NaN scale encoding")
    if mant == 0:
        return 0, 0
    while mant % 2 == 0:
        mant //= 2
        exp += 1
    return mant, exp


def exact_bound(a, w, partial_bound, separable=False):
    """Arbitrary-precision oracle; bound includes ALL prefix sums."""
    if not a or len(a) != len(w) or partial_bound <= 0:
        raise ValueError("nonempty equal group sequences required")
    terms = [(ma * mw, ea + ew) for (ma, ea), (mw, ew) in zip(a, w) if ma * mw]
    if not terms:
        return 0
    anchor = (min(e for m, e in a if m) + min(e for m, e in w if m)
              if separable else min(e for _, e in terms))
    return partial_bound * sum(m << (e - anchor) for m, e in terms)


def inspect_codes(a_codes, w_codes, variant, device="cuda:0", chunk_rows=32):
    """Check every output coordinate, using bounded shifts without overflow.

    Exponents larger than cap are saturated only AFTER choosing the exact
    anchor. Since 2**cap > INT32_MAX/partial_bound, this cannot change a safe/
    unsafe verdict. Do not report the clipped value as an exact maximum bound.
    """
    import torch
    if variant not in PARTIAL_BOUNDS or chunk_rows <= 0:
        raise ValueError("invalid variant/chunk")
    a = a_codes.to(device=device, dtype=torch.long)
    w = w_codes.to(device=device, dtype=torch.long)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1] or a.shape[1] != 32:
        raise ValueError("requires [M,32] and [N,32] source scale codes")
    ak, wk = ("ue8m0", "e4m3") if variant == "o7" else ("e4m3", "e6m2")

    def decode(codes, kind):
        limit = 126 if kind == "e4m3" else 254
        if bool(((codes < 0) | (codes > limit)).any()):
            raise ValueError("source scale contains invalid code")
        lut = torch.tensor([dyadic(c, kind) for c in range(limit + 1)], device=device)
        return lut[codes, 0], lut[codes, 1]

    am, ae = decode(a, ak)
    wm, we = decode(w, wk)
    partial = PARTIAL_BOUNDS[variant]
    threshold = INT32_MAX // partial
    cap = threshold.bit_length()
    totals = {mode: dict(int32_safe_outputs=0, nonzero_outputs=0,
                        safe_with_bound_at_most_2pow24=0, clipped_terms=0,
                        maximum_capped_bound_lower_bound=0)
              for mode in ("tight_output_anchor", "separable_row_column_anchor")}
    spread_hist = torch.zeros(1024, device=device, dtype=torch.long)
    sentinel = 1024
    row_min = torch.where(am > 0, ae, sentinel).amin(-1)
    col_min = torch.where(wm > 0, we, sentinel).amin(-1)
    for start in range(0, a.shape[0], chunk_rows):
        ma = am[start:start + chunk_rows, None, :]
        ea = ae[start:start + chunk_rows, None, :]
        mant = ma * wm[None, :, :]
        exp = ea + we[None, :, :]
        valid = mant > 0
        active = valid.any(-1)
        tight = torch.where(valid, exp, sentinel).amin(-1)
        high = torch.where(valid, exp, -sentinel).amax(-1)
        spread_hist += torch.bincount((high - tight)[active], minlength=1024)
        anchors = {"tight_output_anchor": tight,
                   "separable_row_column_anchor": row_min[start:start + chunk_rows, None] + col_min[None, :]}
        for name, anchor in anchors.items():
            delta = torch.where(valid, exp - anchor[..., None], 0)
            if bool((delta < 0).any()):
                raise AssertionError("anchor would require a lossy right shift")
            # int64: max 15*7*2**17*32 is far below INT64_MAX.
            coeff_sum = (mant << delta.clamp(max=cap)).sum(-1)
            bound = coeff_sum * partial
            r = totals[name]
            r["int32_safe_outputs"] += int((bound <= INT32_MAX).sum())
            r["nonzero_outputs"] += int(active.sum())
            r["safe_with_bound_at_most_2pow24"] += int((bound <= 2**24).sum())
            r["clipped_terms"] += int((valid & (delta > cap)).sum())
            r["maximum_capped_bound_lower_bound"] = max(r["maximum_capped_bound_lower_bound"], int(bound.max()))
    outputs = a.shape[0] * w.shape[0]
    for r in totals.values():
        r.update(outputs=outputs, int32_safe_fraction=r["int32_safe_outputs"] / outputs)
        r["int32_unsafe_bound_outputs"] = outputs - r["int32_safe_outputs"]
    hist = spread_hist.cpu().tolist()
    return dict(variant=variant, partial_bound=partial, source_k_groups=32,
                anchors=totals, exponent_spread_histogram={str(i): n for i, n in enumerate(hist) if n},
                semantics="worst-case reachable-payload triangle bound; failure is NOT observed arithmetic overflow",
                common_real_factor="4*T_W" if variant == "o7" else "T_A/4")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/prepared/llama2_7b_prefill_o0_o4"))
    p.add_argument("--raw-data", type=Path, default=Path("data/raw/llama2_7b_prefill"))
    p.add_argument("--trace-config", type=Path, default=Path("configs/trace/llama2_7b_prefill.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, choices=(4, 24), default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT):
        p.error("fresh output directory inside repository required")
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.trace.storage import sha256_file
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, tensor_identity
    torch.set_num_threads(4)
    started = time.monotonic()
    manifest, prepared_sha = inspect_inputs(args.data)
    raw_manifest, raw_sha = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
    entries = raw_manifest["samples"][:args.samples]
    rows = []
    for entry in entries:
        path = args.raw_data / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("raw file changed")
        raw = _load_and_validate_raw(path, entry["layer"], entry["projection"])
        a, w = [raw[name].to(args.device) for name in ("activation_fp16", "weight_fp16")]
        for variant, (wf, af) in mf.VARIANTS.items():
            aw = mf.quantize_source(a, af)
            ww = mf.quantize_source(w, wf)
            report = inspect_codes(aw["scale"], ww["scale"], variant, args.device)
            report.update(sample_id=entry["sample_id"], raw_file=entry["file"], raw_sha256=entry["sha256"],
                          activation_format=af, weight_format=wf,
                          activation_scale_identity=tensor_identity(aw["scale"]),
                          weight_scale_identity=tensor_identity(ww["scale"]),
                          tensor_scale=float((ww if variant == "o7" else aw)["tensor_scale"].item()))
            rows.append(report)
            print(entry["sample_id"], variant, {k: v["int32_safe_fraction"] for k, v in report["anchors"].items()}, flush=True)
            del aw, ww
        del a, w, raw
    summary = []
    for variant in mf.VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        for anchor in rows[0]["anchors"]:
            s = [r["anchors"][anchor] for r in selected]
            count = sum(r["outputs"] for r in s)
            safe = sum(r["int32_safe_outputs"] for r in s)
            summary.append(dict(variant=variant, anchor=anchor, samples=len(s), outputs=count,
                                int32_safe_outputs=safe, int32_safe_fraction=safe/count,
                                all_outputs_safe_samples=sum(r["int32_safe_outputs"] == r["outputs"] for r in s)))
    if sha256_file(args.data / "manifest.json") != prepared_sha or sha256_file(args.raw_data / "trace_manifest.json") != raw_sha:
        raise ValueError("manifest changed while inspecting")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = dict(scope="read_only_scale_feasibility_no_new_kernel_no_MSE_or_performance_claim",
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  script_sha256=script_sha, prepared_manifest_sha256=prepared_sha, raw_manifest_sha256=raw_sha,
                  gpu=torch.cuda.get_device_name(args.device), torch_version=torch.__version__,
                  inspection_wall_seconds=time.monotonic()-started, summary=summary, samples=rows,
                  caveats=["source-format REAL scale factorization, NOT existing FP32 scale bitwise identity",
                           "bound failure means no data-independent INT32 guarantee; actual partials may be smaller",
                           "passing bound is NOT MSE, safety, throughput, or end-to-end validation",
                           "no CUDA source, extension, prepared trace, quantization payload, or default changed"])
    args.output.mkdir(parents=True)
    (args.output / "feasibility.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

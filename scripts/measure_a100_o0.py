#!/usr/bin/env python3
"""Measure the existing FP16/FP32 cuBLASLt O0 baseline on A100, not raw FP16 trace GEMM."""
import argparse
import json
from pathlib import Path
import statistics
import time

from run_a100_experiment import command, stats

MODES = ("conversion_only", "compute_only", "cold", "steady_state")


def check_payload(p, reference, torch):
    k = dict(p["kernel"])
    assert k["tensor_core"] and k["library"] == "cublasLt"
    assert k["input_dtype"] == "fp16" and k["output_dtype"] == "fp32"
    assert k["compute_type"] == "CUBLAS_COMPUTE_32F" and k["split_k"] <= 1
    assert p["output"].dtype == torch.float32 and torch.isfinite(p["output"]).all()
    for key in ("converted_weight", "converted_activation"):
        torch.testing.assert_close(p[key], reference[key], rtol=0, atol=0)
    torch.testing.assert_close(p["output"], reference["output"], rtol=1e-3, atol=1e-3)
    return float((p["output"] - reference["output"]).abs().max())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--inner", type=int, default=100)
    args = ap.parse_args()
    if args.output.exists() or args.warmup < 0 or args.repeats < 1 or args.inner < 2:
        raise SystemExit("Use a fresh output directory and valid repetition counts")
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared, validate_manifest, sha256_file
    from adangel.reference import run_o0_reference
    from adangel.trace.schema import PreparedInputs
    assert torch.cuda.get_device_capability() == (8, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.data / "manifest.json").read_text())
    validate_manifest(manifest, formal=True, require_arbitrary_bits=True)
    args.output.mkdir(parents=True)
    env = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
               git_commit=command("git", "rev-parse", "HEAD"),
               binary_sha256=sha256_file(Path(native.__file__)),
               warmup=args.warmup, repeats=args.repeats, conversion_inner_repeats=args.inner,
               reference="FP16 dequantized operands cast to FP32; torch matmul TF32 disabled",
               timing="Existing O0 dual timing; conversion-only total batches W+A together")
    (args.output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    validations = []
    # Use the actual experiment shape for long K. For skinny long-K matrices
    # cuBLASLt may offer only split-K heuristics, rejected by the existing O0 contract.
    for m, n, k in ((128,192,256), (4096,4096,4096)):
        for zero in (False, True):
            torch.manual_seed(406)
            a = torch.randint(-127,128,(m,k),device="cuda",dtype=torch.int8)
            if zero:
                a.zero_()
            x = PreparedInputs("synthetic", a, torch.linspace(0.001,0.01,m,device="cuda"),
                               torch.randint(0,256,(n,k//2),device="cuda",dtype=torch.uint8),
                               torch.full((n,k//32),124,device="cuda",dtype=torch.uint8))
            ref = run_o0_reference(x)
            for mode in MODES:
                p = native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,mode,2,3,10)
                error = check_payload(p,ref,torch)
                validations.append(dict(shape=[m,n,k],zero=zero,mode=mode,max_abs_error=error))
    (args.output / "validation.json").write_text(json.dumps(validations, indent=2) + "\n")
    records = []
    for i, sample in enumerate(manifest["samples"]):
        path = args.data / sample["file"]
        assert sha256_file(path) == sample["sha256"]
        x = load_prepared(path,device="cuda")
        ref = run_o0_reference(x)
        with (args.output / "gpu_snapshots.jsonl").open("a") as f:
            f.write(json.dumps(dict(time_unix=time.time(), sample_id=x.sample_id,
                gpu=command("nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv"),
                processes=command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"))) + "\n")
        for mode in MODES:
            p = native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,mode,args.warmup,args.repeats,args.inner)
            error = check_payload(p,ref,torch)
            timings = dict(p["timings_ms"])
            summary = {key: stats(v) for key,v in timings.items()}
            record = dict(sample_id=x.sample_id,variant="o0",mode=mode,timings_ms=timings,
                          summary=summary,kernel=dict(p["kernel"]),timing_method=dict(p["timing_method"]),
                          max_abs_error_vs_reference=error,mse_vs_o0=0.0,
                          stable=all(v["cv_percent"]<3 for v in summary.values()))
            records.append(record)
            with (args.output / "results.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
        print(f"Completed {i+1}/24 {x.sample_id}",flush=True)
    result = dict(modes={},unstable_records=[dict(sample_id=r["sample_id"],mode=r["mode"],
                  bad_stages={k:v["cv_percent"] for k,v in r["summary"].items() if v["cv_percent"]>=3})
                  for r in records if not r["stable"]])
    for mode in MODES:
        rs = [r for r in records if r["mode"]==mode]
        result["modes"][mode] = {stage:dict(median_ms=statistics.median(r["summary"][stage]["median_ms"] for r in rs),
                                mean_ms=statistics.fmean(r["summary"][stage]["median_ms"] for r in rs))
                                for stage in rs[0]["summary"]}
    (args.output / "summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

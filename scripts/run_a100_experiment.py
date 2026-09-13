#!/usr/bin/env python3
"""Validate/run SM80 O1 and O3 with native O0 as the MSE reference."""
import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path


def stats(v):
    values = [float(x) for x in v]
    mean = statistics.fmean(values)
    return dict(count=len(values), median_ms=statistics.median(values), mean_ms=mean,
                cv_percent=statistics.pstdev(values)/mean*100 if mean else 0,
                min_ms=min(values), max_ms=max(values))


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def validate(native, torch):
    from adangel.quantization.mxfp4 import (
        decode_ue8m0_tensor, mxfp4_to_q4_packed, unpack_int4_tensor,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = []
    for k in (128, 256, 384):
        for pattern in ("zero", "codes_and_saturation", "random"):
            torch.manual_seed(174+k)
            a = torch.randint(-128,128,(64,k),dtype=torch.int8,device="cuda")
            w = torch.randint(0,256,(64,k//2),dtype=torch.uint8,device="cuda")
            if pattern == "zero":
                a.zero_()
            if pattern == "codes_and_saturation":
                a[:] = (torch.arange(k,device="cuda")%256-128).to(torch.int8)
                w[:] = (torch.arange(k//2,device="cuda")%256).to(torch.uint8)
            asc = torch.linspace(0.002,0.08,64,device="cuda")
            for variant, g in (("o1",32),("o3",128)):
                ws = (torch.arange(64*(k//g),device="cuda").reshape(64,k//g)%7+123).to(torch.uint8)
                if variant == "o3":
                    weight = unpack_int4_tensor(mxfp4_to_q4_packed(w)).float()
                    factor = 1.0
                else:
                    nib = torch.stack((w&15,w>>4),dim=-1).reshape(64,k).long()
                    lut=torch.tensor([0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12],device="cuda")
                    weight=lut[nib].float()
                    factor=0.5
                scale=decode_ue8m0_tensor(ws)
                reference=torch.zeros((64,64),device="cuda")
                for group in range(k//g):
                    sl=slice(group*g,(group+1)*g)
                    partial=a[:,sl].float()@weight[:,sl].T
                    reference.add_(partial*(asc[:,None]*(scale[:,group][None,:]*factor)))
                for mode in ("conversion_only","compute_only","cold","steady_state"):
                    p=native.benchmark(variant,mode,a,asc,w,ws,2,3,10)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(p["output"],reference,rtol=1e-3,atol=1e-3)
                    assert torch.isfinite(p["output"]).all()
                    checks.append(dict(variant=variant,k=k,pattern=pattern,mode=mode,
                        max_abs_error=float((p["output"]-reference).abs().max())))
    return dict(passed=True,checks=checks)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--validate-only",action="store_true")
    ap.add_argument("--warmup",type=int,default=50)
    ap.add_argument("--repeats",type=int,default=200)
    ap.add_argument("--inner",type=int,default=100)
    ap.add_argument("--samples",type=int,default=0)
    args=ap.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; use a fresh run directory")
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,validate_manifest,sha256_file
    assert torch.cuda.get_device_capability()==(8,0)
    args.output.mkdir(parents=True)
    env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(),git_commit=command("git","rev-parse","HEAD"),
             git_status=command("git","status","--porcelain"),
             driver=command("nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"),
             nvcc=command("nvcc","--version"),warmup=args.warmup,repeats=args.repeats,
             conversion_inner_repeats=args.inner,
             timing="amortized isolated conversion; direct single-path GEMM/cold/steady",
             clock_policy="unlocked; no external GPU processes modified")
    (args.output/"environment.json").write_text(json.dumps(env,indent=2)+"\n")
    validation=validate(native,torch)
    (args.output/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print("Synthetic correctness:",len(validation["checks"]),"checks passed",flush=True)
    if args.validate_only:
        return
    if not args.data:
        raise SystemExit("--data required for real-trace experiment")
    manifest=json.loads((args.data/"manifest.json").read_text())
    validate_manifest(manifest,formal=True,require_arbitrary_bits=True)
    (args.output/"data_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    samples=manifest["samples"][:args.samples or None]
    results=[]
    for i,sample in enumerate(samples):
        path=args.data/sample["file"]
        assert sha256_file(path)==sample["sha256"],f"SHA mismatch: {path}"
        x=load_prepared(path,device="cuda")
        o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,"compute_only",5,10,args.inner)
        reference=o0["output"]
        assert reference.dtype==torch.float32 and torch.isfinite(reference).all()
        for j,mode in enumerate(("conversion_only","compute_only","cold","steady_state")):
            order=("o1","o3") if (i+j)%2==0 else ("o3","o1")
            for variant in order:
                w,ws=(x.W_mxfp4,x.W_scale) if variant=="o1" else (x.W_mxfp4_g128,x.W_scale_g128)
                p=native.benchmark(variant,mode,x.A_int8,x.A_scale,w,ws,args.warmup,args.repeats,args.inner)
                torch.cuda.synchronize()
                assert p["output"].dtype==torch.float32 and torch.isfinite(p["output"]).all()
                mse=float((p["output"].double()-reference.double()).square().mean())
                summary={key:stats(v) for key,v in p["timings_ms"].items()}
                record=dict(sample_id=x.sample_id,variant=variant,mode=mode,shape=list(x.shape),
                    timings_ms=dict(p["timings_ms"]),summary=summary,mse_vs_o0=mse,
                    kernel=dict(p["kernel"]),o0_kernel=dict(o0["kernel"]),
                    stable=all(s["cv_percent"]<3 for s in summary.values()))
                results.append(record)
                with (args.output/"results.jsonl").open("a") as f:
                    f.write(json.dumps(record)+"\n")
        print(f"Completed {i+1}/{len(samples)}: {x.sample_id}",flush=True)
    summary={}
    for variant in ("o1","o3"):
        by_mode={}
        for mode in ("conversion_only","compute_only","cold","steady_state"):
            rs=[r for r in results if r["variant"]==variant and r["mode"]==mode]
            by_mode[mode]={stage:dict(median_ms=statistics.median([r["summary"][stage]["median_ms"] for r in rs]),
                mean_ms=statistics.fmean([r["summary"][stage]["median_ms"] for r in rs])) for stage in rs[0]["summary"]}
        mses=[r["mse_vs_o0"] for r in results if r["variant"]==variant and r["mode"]=="compute_only"]
        summary[variant]=dict(modes=by_mode,mse_median=statistics.median(mses),mse_mean=statistics.fmean(mses))
    bykey={(r["sample_id"],r["variant"],r["mode"]):r for r in results}
    ratios=[bykey[(s["sample_id"],"o1","compute_only")]["summary"]["gemm"]["median_ms"]/
            bykey[(s["sample_id"],"o3","compute_only")]["summary"]["gemm"]["median_ms"] for s in samples]
    summary["o3_throughput_over_o1_paired_median"]=statistics.median(ratios)
    summary["unstable_records"]=[dict(sample_id=r["sample_id"],variant=r["variant"],mode=r["mode"],
        bad_stages={k:s["cv_percent"] for k,s in r["summary"].items() if s["cv_percent"]>=3}) for r in results if not r["stable"]]
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()

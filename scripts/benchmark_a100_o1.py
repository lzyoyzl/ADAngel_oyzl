#!/usr/bin/env python3
"""SM80 O1 candidate correctness and paired, unfiltered CUDA-event measurement."""
import argparse
import json
from pathlib import Path
import statistics
import time

from run_a100_experiment import command, stats as basic_stats


def stats(values):
    import numpy as np
    result=basic_stats(values)
    q=np.percentile(values,[5,25,75,95])
    result.update(p5_ms=float(q[0]),p95_ms=float(q[3]),iqr_ms=float(q[2]-q[1]))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--impl', nargs='+', default=['baseline', 'swizzle_64x64_k128',
                   'swizzle_128x64_k128', 'swizzle_128x128_k128', 'swizzle_128x64_k64'])
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--all-modes', action='store_true')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--profile', action='store_true', help='Single target call for NCU; no reference kernels')
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Use a fresh output directory')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared, sha256_file, validate_manifest
    assert torch.cuda.get_device_capability() == (8, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    def save(name, obj):
        (args.output/name).write_text(json.dumps(obj, indent=2)+'\n')
    def append(name, obj):
        with (args.output/name).open('a') as f:
            f.write(json.dumps(obj)+'\n')
    save('environment.json', dict(git_commit=command('git', 'rev-parse', 'HEAD'),
         git_status=command('git', 'status', '--porcelain'), torch=torch.__version__,
         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
         binary_sha256=sha256_file(Path(native.__file__)), args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         policy='No filtering; unlocked clocks; per-round cyclic/reversed order; conversion amortized, total directly timed'))
    if args.validate or args.validate_only:
        checks = []
        for m,n,k in [(128,128,128),(256,128,256),(128,256,384),(128,128,4096)]:
            for pattern in ['zero','random','saturation','scale_codes']:
                torch.manual_seed(718+k)
                a=torch.randint(-128,128,(m,k),dtype=torch.int8,device='cuda')
                w=torch.randint(0,256,(n,k//2),dtype=torch.uint8,device='cuda')
                if pattern=='zero': a.zero_()
                if pattern=='saturation': a[:]=(torch.arange(k,device='cuda')%256-128).to(torch.int8)
                asc=torch.linspace(.0001,.1,m,device='cuda')
                ws=(torch.arange(n*(k//32),device='cuda').reshape(n,k//32)%17+116).to(torch.uint8)
                if pattern=='scale_codes':
                    ws=(torch.arange(n*(k//32),device='cuda').reshape(n,k//32)%255).to(torch.uint8)
                    asc=torch.linspace(1e-7,1e-6,m,device='cuda')
                base=native.benchmark('o1','compute_only',a,asc,w,ws,0,1,1,'baseline')['output']
                for impl in args.impl:
                    for mode in ['conversion_only','compute_only','cold','steady_state']:
                        out=native.benchmark('o1',mode,a,asc,w,ws,0,1,1,impl)['output']
                        assert torch.isfinite(out).all() and out.dtype==torch.float32
                        torch.testing.assert_close(out,base,rtol=0,atol=0)
                        assert torch.equal(out.view(torch.int32),base.view(torch.int32))
                        checks.append(dict(shape=[m,n,k],pattern=pattern,implementation=impl,mode=mode,bitwise_equal=True))
        save('validation.json',dict(passed=True,checks=checks))
        print(f'Synthetic bitwise checks passed: {len(checks)}',flush=True)
        for impl in args.impl:
            for bad in ['code255','negative_a_scale','nan_a_scale']:
                bad_ws=ws.clone();bad_as=asc.clone()
                if bad=='code255': bad_ws[0,0]=255
                if bad=='negative_a_scale': bad_as[0]=-1
                if bad=='nan_a_scale': bad_as[0]=float('nan')
                try:
                    native.benchmark('o1','compute_only',a,bad_as,w,bad_ws,0,1,1,impl)
                except RuntimeError:
                    checks.append(dict(implementation=impl,rejected=bad))
                else:
                    raise AssertionError(f'{impl} accepted {bad}')
        save('validation.json',dict(passed=True,checks=checks))
        if args.validate_only: return
    manifest=json.loads((args.data/'manifest.json').read_text())
    validate_manifest(manifest,formal=True,require_arbitrary_bits=True)
    save('data_manifest.json',manifest)
    summaries=[]
    modes=['conversion_only','compute_only','cold','steady_state'] if args.all_modes else ['compute_only']
    for i,entry in enumerate(manifest['samples'][:args.samples or None]):
        path=args.data/entry['file']
        assert sha256_file(path)==entry['sha256']
        x=load_prepared(path,device='cuda')
        def call(impl,mode):
            if impl=='o0':
                return native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,mode,args.warmup,args.repeats,args.inner)
            return native.benchmark('o1',mode,x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,args.warmup,args.repeats,args.inner,impl)
        if args.profile:
            for impl in args.impl: call(impl,'compute_only')
            return
        base=native.benchmark('o1','compute_only',x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,0,1,1,'baseline')['output']
        ref=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,100)['output']
        baseline_mse=float((base.double()-ref.double()).square().mean())
        append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,time=time.time(),
            gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv'),
            processes=command('nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv')))
        for mode in modes:
            collected={v:{} for v in ['o0']+args.impl}
            ratios={v:[] for v in args.impl}
            for r in range(args.rounds):
                order=list(collected)
                shift=(i+r)%len(order)
                order=order[shift:]+order[:shift]
                if r%2: order.reverse()
                medians={}
                for impl in order:
                    out=call(impl,mode)
                    y=out['output']
                    assert torch.isfinite(y).all()
                    if impl!='o0':
                        torch.testing.assert_close(y,base,rtol=0,atol=0)
                        assert torch.equal(y.view(torch.int32),base.view(torch.int32))
                    mse=float((y.double()-ref.double()).square().mean())
                    st={k:stats(v) for k,v in out['timings_ms'].items()}
                    for key,vals in out['timings_ms'].items(): collected[impl].setdefault(key,[]).extend(vals)
                    metric='gemm' if mode=='compute_only' else 'total'
                    medians[impl]=st[metric]['median_ms']
                    append('results.jsonl',dict(sample_id=x.sample_id,implementation=impl,mode=mode,round=r,
                        kernel=dict(out['kernel']),timings_ms=dict(out['timings_ms']),summary=st,mse_vs_o0=mse,
                        baseline_mse_vs_o0=baseline_mse,bitwise_equal_baseline=impl!='o0'))
                for impl in args.impl: ratios[impl].append(medians['o0']/medians[impl])
            result=dict(sample_id=x.sample_id,mode=mode,baseline_mse_vs_o0=baseline_mse,
                timings={v:{s:stats(vals) for s,vals in stages.items()} for v,stages in collected.items()},
                paired_speedups_vs_o0={v:statistics.median(vals) for v,vals in ratios.items()},round_speedups_vs_o0=ratios)
            summaries.append(result)
            print(json.dumps(result),flush=True)
            save('summary.json',dict(samples=summaries))


if __name__=='__main__': main()

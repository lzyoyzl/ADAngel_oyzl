#!/usr/bin/env python3
"""A100 O3 optimization: exact old-O3 comparison, MSE vs O0, paired current O1."""
import argparse
import json
from pathlib import Path
import time
from benchmark_a100_o1 import stats, command


def expected_o3_production(m,n,k):
    if m%64==0 and n%128==0 and k%256==0:
        return 'o3_swizzle_64x128_k256_exp_static_stream_bound2_store2'
    return 'baseline'


def check_o3_production_metadata(result,m,n,k):
    kernel=dict(result['kernel'])
    expected=expected_o3_production(m,n,k)
    assert kernel['requested_implementation']=='production'
    assert kernel['implementation']==expected,(kernel,expected)
    assert kernel['production_shape_fallback']==(expected=='baseline')
    if expected!='baseline':
        assert list(kernel['cta_tile'])==[64,128,256]
        assert kernel['kernel_symbol']=='adangel_sm80_o3_swizzled_bound2'
        assert kernel['output_store_bits']==64 and kernel['scale_load_bits']==8


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--impl',nargs='+',default=['o3_swizzle_64x64_k128_magic','o3_swizzle_128x64_k128_magic','o3_swizzle_128x64_k256_magic'])
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--inner',type=int,default=100)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--all-modes',action='store_true')
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('Use a fresh output directory')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import load_prepared,sha256_file,validate_manifest
    assert torch.cuda.get_device_capability()==(8,0)
    args.output.mkdir(parents=True)
    def save(name,data): (args.output/name).write_text(json.dumps(data,indent=2)+'\n')
    def append(name,data):
        with (args.output/name).open('a') as f: f.write(json.dumps(data)+'\n')
    save('environment.json',dict(commit=command('git','rev-parse','HEAD'),binary_sha256=sha256_file(Path(native.__file__)),
        cuda_sources_sha256={name:sha256_file(Path(__file__).resolve().parents[1]/name) for name in
            ('csrc/sm80/o1_o3.cu','csrc/sm80/o1_optimized.cuh','csrc/sm80/o3_optimized.cuh')},
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        policy='No filtering, unlocked clocks, same-process alternating order; current production O1 reference'))
    if args.validate or args.validate_only:
        checks=[]
        for m,n,k in [(128,128,256),(128,192,512),(256,128,4096),(64,64,128),
                      (64,128,128),(64,128,384),(64,128,768)]:
            for pattern in ['zero','random','saturation','scale_codes','zero_scale']:
                torch.manual_seed(832+k)
                a=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
                w=torch.randint(0,256,(n,k//2),device='cuda',dtype=torch.uint8)
                if pattern=='zero': a.zero_()
                if pattern=='saturation': a[:]=(torch.arange(k,device='cuda')%256-128).to(torch.int8)
                asc=torch.linspace(.0001,.1,m,device='cuda')
                if pattern=='zero_scale': asc.zero_()
                ws=(torch.arange(n*(k//128),device='cuda').reshape(n,k//128)%17+116).to(torch.uint8)
                if pattern=='scale_codes':
                    ws=(torch.arange(n*(k//128),device='cuda').reshape(n,k//128)%255).to(torch.uint8)
                    asc=torch.linspace(1e-7,1e-6,m,device='cuda')
                base=native.benchmark('o3','compute_only',a,asc,w,ws,0,1,1,'baseline')['output']
                for impl in args.impl:
                    if '128x64' in impl and m%128: continue
                    if 'x128_' in impl and n%128: continue
                    if 'k256' in impl and k%256: continue
                    for mode in ['conversion_only','compute_only','cold','steady_state']:
                        r=native.benchmark('o3',mode,a,asc,w,ws,0,1,1,impl)
                        if impl=='production':check_o3_production_metadata(r,m,n,k)
                        y=r['output']
                        assert y.dtype==torch.float32 and torch.isfinite(y).all()
                        torch.testing.assert_close(y,base,rtol=0,atol=0)
                        assert torch.equal(y.view(torch.int32),base.view(torch.int32))
                        checks.append(dict(implementation=impl,shape=[m,n,k],pattern=pattern,mode=mode,bitwise_equal=True))
        rejected=[]
        for impl in args.impl:
            a=torch.ones((128,128),device='cuda',dtype=torch.int8)
            asc=torch.ones(128,device='cuda');w=torch.zeros((128,64),device='cuda',dtype=torch.uint8)
            ws=torch.full((128,1),127,device='cuda',dtype=torch.uint8)
            # K256 candidates need their native alignment before validation.
            if 'k256' in impl:
                a=a.repeat(1,2);w=w.repeat(1,2);ws=ws.repeat(1,2)
            for bad in ('scale_255','negative_a_scale','nan_a_scale'):
                bad_as=asc.clone();bad_ws=ws.clone()
                if bad=='scale_255': bad_ws[0,0]=255
                elif bad=='negative_a_scale': bad_as[0]=-1
                else: bad_as[0]=float('nan')
                try:native.benchmark('o3','compute_only',a,bad_as,w,bad_ws,0,1,1,impl)
                except RuntimeError as error:
                    expected='UE8M0 code 255' if bad=='scale_255' else 'invalid activation scale'
                    assert expected in str(error),str(error)
                    rejected.append(dict(implementation=impl,invalid_input=bad,rejected=True))
                else: raise AssertionError((impl,bad,'invalid input accepted'))
        save('validation.json',dict(passed=True,checks=checks,rejected_inputs=rejected))
        print('Exact O3 checks:',len(checks),flush=True)
        if args.validate_only: return
    manifest=json.loads((args.data/'manifest.json').read_text())
    validate_manifest(manifest,formal=True,require_arbitrary_bits=True)
    save('data_manifest.json',manifest)
    modes=['conversion_only','compute_only','cold','steady_state'] if args.all_modes else ['compute_only']
    for i,entry in enumerate(manifest['samples'][:args.samples or None]):
        path=args.data/entry['file'];assert sha256_file(path)==entry['sha256']
        x=load_prepared(path,device='cuda')
        def call(impl,mode):
            if impl=='o0': return native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,mode,args.warmup,args.repeats,args.inner)
            if impl=='o1': return native.benchmark('o1',mode,x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,args.warmup,args.repeats,args.inner,'production')
            result=native.benchmark('o3',mode,x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,args.warmup,args.repeats,args.inner,impl)
            if impl=='production':check_o3_production_metadata(result,x.A_int8.shape[0],x.W_mxfp4_g128.shape[0],x.A_int8.shape[1])
            return result
        if args.profile:
            for impl in args.impl: call(impl,'compute_only')
            return
        reference=call('o0','compute_only')['output']
        base=call('baseline','compute_only')['output']
        baseline_mse=float((base.double()-reference.double()).square().mean())
        append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,time=time.time(),
            gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
        for mode in modes:
            for r in range(args.rounds):
                order=['o0','o1','baseline']+args.impl
                shift=(i+r)%len(order);order=order[shift:]+order[:shift]
                if r%2:order.reverse()
                for impl in order:
                    result=call(impl,mode); y=result['output']
                    assert torch.isfinite(y).all()
                    if impl not in ('o0','o1'):
                        assert torch.equal(y.view(torch.int32),base.view(torch.int32)),(x.sample_id,impl)
                    mse=float((y.double()-reference.double()).square().mean())
                    append('results.jsonl',dict(sample_id=x.sample_id,mode=mode,round=r,implementation=impl,
                        timings_ms=dict(result['timings_ms']),summary={s:stats(v) for s,v in result['timings_ms'].items()},
                        kernel=dict(result['kernel']),mse_vs_o0=mse,baseline_mse_vs_o0=baseline_mse,
                        bitwise_equal_baseline=impl not in ('o0','o1')))
            print(x.sample_id,mode,'done',flush=True)


if __name__=='__main__': main()

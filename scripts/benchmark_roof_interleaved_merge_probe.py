#!/usr/bin/env python3
"""v63 prepared-core validation and paired screen; conversion/default unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from benchmark_roof_m128_probe import Driver as M128Driver, summary as m128_summary
from benchmark_a100_o1 import command, stats
from benchmark_a100_roof_trace import measurement_order
from benchmark_a100_roof_candidates import group_major_scales
from benchmark_a100_mixed import validate_fp16_result
from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, mse, source_identity, verify_raw_prepared
from probe_roof_interleaved_merge_codegen import ROOT, HEADERS, generated_header
from roof_payload_validation import verify_grouped_payload
from roof_reduction_validation import reference_fp64


def checked_cubins(directory):
    x = json.loads((directory/'codegen.json').read_text())
    if not x['control_comparison']['passed']:
        raise ValueError('control SASS differs from best54/59')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(ROOT/'csrc/sm80/roof_m128_probe.cu') != x['wrapper_sha256']:
        raise ValueError('wrapper source drift')
    paths = {}
    for policy in (0,1):
        for family,name in HEADERS.items():
            entry=x['generated_headers'][f'{family}_{policy}']
            source=ROOT/'csrc/sm80'/name
            generated=directory/entry['generated']
            if (digest(source)!=entry['source_sha256'] or digest(generated)!=entry['generated_sha256'] or
                generated.read_text()!=generated_header(source.read_text(),policy)):
                raise ValueError('generated source drift')
        path=(directory/f'interleaved_merge_{policy}.cubin').resolve()
        v=x['variants'][str(policy)]
        if digest(path)!=v['cubin_sha256']:
            raise ValueError('cubin drift')
        if set(v['entries'])!={'adangel_roof_m128_o3','adangel_roof_m128_o78'} or not all(
            e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'] and not e['int8_mma']
            for e in v['entries'].values()):
            raise ValueError('same-entry ISA audit failed')
        paths[policy]=path
    return paths


class Driver:
    """Reuse the tested native Event launcher with tile_m=64 for BOTH policies."""
    def __init__(self, library, cubins, variant, smem):
        self.drivers={};self.resources={}
        try:
            for policy,path in cubins.items():
                driver=M128Driver(library,{0:path},variant,smem)
                self.drivers[policy]=driver
                info=dict(driver.resources[0]);info.pop('m128')
                info['interleaved_merge']=policy
                info['partial_registers_per_four_n_atoms']=16 if policy else 32
                self.resources[policy]=info
        except Exception:
            self.close();raise

    def run(self,policy,*args): return self.drivers[policy].run(0,*args)

    def close(self):
        for driver in self.drivers.values(): driver.close()
        self.drivers.clear()


def summary(rows):
    # Identical pairing/bootstrap/MSE gates, with an explicitly renamed key.
    result=m128_summary([dict(r,m128=r['interleaved_merge']) for r in rows])
    for row in result: row['interleaved_merge']=row.pop('m128')
    return result


def validate(library,cubins,variants):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    checks=[]
    for variant in variants:
        for m,n,k in ((64,128,128),(64,128,384),(128,256,640),(64,128,4096)):
            for pattern in ('random','zero','extrema','zero_scale'):
                torch.manual_seed(20261002+k)
                low,high=(-32,32) if variant=='o8' else (-128,128)
                act=torch.randint(low,high,(m,k),device='cuda',dtype=torch.int8)
                weight=torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int8)
                if pattern=='zero': act.zero_();weight.zero_()
                if pattern=='extrema':
                    act[:,::2]=low;act[:,1::2]=high-1
                    weight[:,::2]=-8;weight[:,1::2]=7
                g=k//128
                if variant=='o3':
                    asc=torch.linspace(.001,.03,m,device='cuda')
                    wsc=((torch.arange(n*g,device='cuda').reshape(n,g)*7)%13+116).byte()
                    if pattern=='zero_scale': asc.zero_()
                    tune=54
                else:
                    def scales(rows,factor):
                        row=torch.arange(rows,device='cuda')[:,None]
                        group=torch.arange(g,device='cuda')[None,:]
                        return group_major_scales((1+(row*factor+group*29)%113/128)*
                            torch.exp2(((row+3*group)%7-10).float()))
                    asc,wsc=scales(m,13),scales(n,17)
                    if pattern=='zero_scale': asc[:,::2]=0;wsc[:,1::2]=0
                    tune=59
                values=(split_int8_to_packed_int4(act),asc,pack_q4(weight),wsc)
                semantic=reference_fp64(variant,values)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    best=native._benchmark_roof_candidate(variant,tune,*values,0,1)
                    verify_grouped_payload(best,tune,values[0],values[2],values[3])
                    driver=Driver(library,cubins,variant,best['kernel']['shared_memory_bytes'])
                    try:
                        for policy in (0,1):
                            ws=best['converted_weight_scale'] if variant=='o3' else wsc
                            y,_=driver.run(policy,best,asc,ws,0,1)
                            torch.testing.assert_close(y.double(),semantic,rtol=1e-3,atol=1e-3)
                            checks.append(dict(variant=variant,shape=[m,n,k],pattern=pattern,policy=policy,
                                bitwise_equal_best=True,finite_fp32=True,probe_resources=driver.resources[policy],
                                max_abs_error_fp64=(y.double()-semantic).abs().max().item()))
                    finally: driver.close()
                stream.synchronize()
        print(variant,'synthetic validation passed',flush=True)
    return dict(passed=True,count=len(checks),checks=checks,
                scope='prepared_core_small_MN_variable_K_including4096_nondefault_stream')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--variants',nargs='+',choices=['o3','o7','o8'],default=['o3','o7','o8'])
    args=p.parse_args()
    if (args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or not 1<=args.samples<=24 or
        min(args.rounds,args.repeats)<1 or args.warmup<0 or len(set(args.variants))!=len(args.variants)):
        p.error('fresh repository output and valid measurement coverage required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    cubins=checked_cubins(args.cubins)
    args.output.mkdir(parents=True)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    library=(args.output/'libroof_probe_driver.so').resolve()
    build=['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
        str(ROOT/'csrc/sm80/roof_m128_driver.cpp'),'-lcuda','-o',str(library)]
    with (args.output/'driver_build.log').open('w') as log:
        subprocess.run(build,stdout=log,stderr=subprocess.STDOUT,check=True)
    env=dict(git_commit=command('git','rev-parse','HEAD'),extension_sha256=sha256_file(Path(native.__file__)),
        cubins={str(i):sha256_file(path) for i,path in cubins.items()},driver_sha256=sha256_file(library),
        driver_build_command=build,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='isolated_prepared_core_compute_only_not_conversion_or_four_mode_acceptance',
        policy='unlocked shared GPU; cyclic paired order; no filtering',
        timing='native_driver_single_launch_CUDA_Event_allocation_outside_interval')
    save('environment.json',env)
    save('validation.json',validate(library,cubins,args.variants))
    if args.validate_only: return
    manifest,mh=inspect_inputs(args.data)
    raw_manifest,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    raw_entries={r['sample_id']:r for r in raw_manifest['samples']}
    env.update(prepared_manifest_sha256=mh,raw_manifest_sha256=rh);save('environment.json',env)
    rows=[]
    for si,entry in enumerate(manifest['samples'][:args.samples]):
        path=args.data/entry['file'];re=raw_entries[entry['sample_id']];rp=args.raw_data/re['file']
        assert sha256_file(path)==entry['sha256'] and sha256_file(rp)==re['sha256']
        x=load_prepared(path,device='cuda');raw=_load_and_validate_raw(rp,re['layer'],re['projection'])
        operands=(raw['activation_fp16'],raw['weight_fp16']);verify_raw_prepared(x,operands)
        assert x.sample_id==entry['sample_id'] and list(x.shape)==entry['shape']
        o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,100)['output']
        for vi,variant in enumerate(args.variants):
            if variant=='o3':
                base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,100,'production')
                values=(base['converted_activation'],x.A_scale,base['converted_weight'],x.W_scale_g128)
                paired=o0;ref='o0';tune=54
            else:
                wf,af=mf.VARIANTS[variant]
                wsrc=mf.quantize_source(operands[1].cuda(),wf);asrc=mf.quantize_source(operands[0].cuda(),af)
                append('source_provenance.jsonl',dict(sample_id=x.sample_id,variant=variant,
                    raw_prepared_replay_bitwise=True,weight=source_identity(wsrc),activation=source_identity(asrc)))
                ref=mf.PAIRED_BASELINE[variant]
                fp=native._benchmark_mixed(ref,'compute_only',wsrc,asrc,0,1,100,'64x128x256','row_major')
                validate_fp16_result(fp,wsrc,asrc);paired=fp['output']
                base=native._benchmark_mixed(variant,'compute_only',wsrc,asrc,0,1,100,'64x128x256','group_major')
                values=(*base['converted_activation'],*base['converted_weight']);tune=59
            best=native._benchmark_roof_candidate(variant,tune,*values,0,1)
            payload=verify_grouped_payload(best,tune,values[0],values[2],values[3])
            scale=best['converted_weight_scale'] if variant=='o3' else values[3]
            driver=Driver(library,cubins,variant,best['kernel']['shared_memory_bytes'])
            try:
                for r in range(args.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=r,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order((0,1),si,vi,r)
                    for policy in order:
                        y,times=driver.run(policy,best,values[1],scale,args.warmup,args.repeats)
                        row=dict(sample_id=x.sample_id,variant=variant,round=r,interleaved_merge=policy,execution_order=order,
                            raw_ms=times,summary=stats(times),bitwise_equal_current_best=True,reference_tune=tune,
                            mse_vs_current_best=0.0,mse_vs_paired_fp16=mse(y,paired),mse_vs_o0=mse(y,o0),paired_fp16=ref,
                            reference_kernel=dict(best['kernel']),probe_resources=driver.resources[policy],**payload,
                            probe_symbol='adangel_roof_m128_o3' if variant=='o3' else 'adangel_roof_m128_o78')
                        rows.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,'paired measurement complete',flush=True)
            finally: driver.close()
    assert len(rows)==args.samples*len(args.variants)*args.rounds*2
    save('summary.json',dict(correctness_passed=True,no_filtering=True,production_default_changed=False,
        scope='internal_cubin_compute_only_screen',records=summary(rows)))
    print(json.dumps(summary(rows),indent=2),flush=True)


if __name__=='__main__': main()

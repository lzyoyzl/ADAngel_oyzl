#!/usr/bin/env python3
"""Same-GEMM54 O3 conversion candidates; original24 inputs and native four modes."""
import argparse
import json
from pathlib import Path
import time

from benchmark_a100_o1 import command,stats
from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,mse
from benchmark_conversion_pipeline import summarize
from benchmark_integer_conversion_probe import order
from validate_o3_conversion_pipeline import validate_pair,MODES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--samples',type=int,default=24);p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    p.add_argument('--implementations',type=int,nargs='+',default=[0,1,2])
    args=p.parse_args()
    if (args.output.exists() or not 1<=args.samples<=24 or args.rounds<1 or args.warmup<0 or args.repeats<2
        or args.inner<2 or 0 not in args.implementations or len(set(args.implementations))!=len(args.implementations)
        or any(i not in (0,1,2) for i in args.implementations)):
        p.error('fresh output, baseline0, unique valid implementations and positive counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('A100 required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,mh=inspect_inputs(args.data)
    raw_manifest,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    raw_index={e['sample_id']:e for e in raw_manifest['samples']}
    args.output.mkdir(parents=True)
    def save(name,obj): (args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def append(name,obj):
        with (args.output/name).open('a') as f: f.write(json.dumps(obj,allow_nan=False)+'\n')
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),binary_sha256=sha256_file(Path(native.__file__)),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gemm_tune=54,gemm_math_changed=False,
        prepared_manifest_sha256=mh,raw_manifest_sha256=rh,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        policy='shared GPU; unlocked; no filtering; cyclic order',timing_contract_version=2,
        input_policy='unchanged_prepared_O3_replayed_against_original_FP16'))
    records=[]
    for si,e in enumerate(manifest['samples'][:args.samples]):
        re=raw_index[e['sample_id']]
        if sha256_file(args.data/e['file'])!=e['sha256'] or sha256_file(args.raw_data/re['file'])!=re['sha256']:
            raise ValueError('input changed after preflight')
        raw=_load_and_validate_raw(args.raw_data/re['file'],re['layer'],re['projection'])
        x=load_prepared(args.data/e['file'],device='cuda')
        verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']))
        append('source_provenance.jsonl',dict(sample_id=e['sample_id'],prepared_sha256=e['sha256'],raw_sha256=re['sha256'],replay_bitwise=True))
        base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,0,1,2,'production',54,0)
        o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,2)['output']
        error=mse(base['output'],o0)
        for ri in range(args.rounds):
            append('gpu_snapshots.jsonl',dict(sample_id=e['sample_id'],round=ri,time=time.time(),
                gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
            for mi,mode in enumerate(MODES):
                for impl in order(args.implementations,si,mi,ri):
                    result=native.benchmark('o3',mode,x.A_int8,x.A_scale,x.W_mxfp4_g128,x.W_scale_g128,
                        args.warmup,args.repeats,args.inner,'production',54,impl)
                    validate_pair(result,base,mode,impl,args.inner,args.repeats)
                    values={k:list(v) for k,v in result['timings_ms'].items()}
                    summaries={k:stats(v) for k,v in values.items()}
                    selected='gemm' if mode=='compute_only' else 'total'
                    row=dict(sample_id=e['sample_id'],variant='o3',mode=mode,round=ri,implementation=impl,
                        raw_ms=values,stage_summaries=summaries,summary=summaries[selected],selected_stage=selected,
                        total_timing=result['total_timing'],stage_timing_inner_repeats=dict(result['stage_timing_inner_repeats']),
                        weight_cached=result['weight_cached'],activation_prepared=result['activation_prepared'],
                        kernel=dict(result['kernel']),bitwise_equal_current_best=True,mse_vs_current_best=0.,
                        mse_vs_paired_fp16=error,paired_reference='o0')
                    records.append(row);append('results.jsonl',row)
                    print(e['sample_id'],mode,impl,summaries[selected]['median_ms'],flush=True)
        del base,o0,result,x,raw
        save('summary.json',summarize(records))
        save('validation.json',dict(passed=True,records=len(records),samples=si+1,output_bitwise_current_best=True))
    print('O3 CONVERSION FOUR MODES PASSED',flush=True)


if __name__=='__main__': main()

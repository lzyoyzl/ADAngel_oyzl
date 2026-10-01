#!/usr/bin/env python3
"""v35 same-GEMM59 conversion AB, four native timing modes, original24 traces."""
import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command,stats
from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,source_identity,mse
from benchmark_integer_conversion_probe import order
from validate_conversion_pipeline import validate_pair


def summarize(rows):
    from adangel.benchmark.metrics import bootstrap_median_ci
    index={(r['sample_id'],r['variant'],r['mode'],r['round'],r['implementation']):r for r in rows}
    if len(index)!=len(rows): raise ValueError('duplicate record')
    summary=[]
    for v,mode,impl in sorted({(r['variant'],r['mode'],r['implementation']) for r in rows}):
        subset=[r for r in rows if (r['variant'],r['mode'],r['implementation'])==(v,mode,impl)]
        samples=[]
        for sid in sorted({r['sample_id'] for r in subset}):
            group=[r for r in subset if r['sample_id']==sid]
            ratios=[index[(sid,v,mode,r['round'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in group]
            samples.append(dict(sample_id=sid,median_ms=statistics.median(r['summary']['median_ms'] for r in group),
                paired_speedup=statistics.median(ratios),mse_vs_paired_fp16=group[0]['mse_vs_paired_fp16']))
        ratios=[s['paired_speedup'] for s in samples]
        summary.append(dict(variant=v,mode=mode,implementation=impl,samples=len(samples),records=len(subset),
            median_ms=statistics.median(s['median_ms'] for s in samples),
            paired_speedup=statistics.median(ratios),
            paired_speedup_ci95=list(bootstrap_median_ci(ratios,10000,.95,20261001)) if len(samples)>1 else None,
            median_mse_vs_paired_fp16=statistics.median(s['mse_vs_paired_fp16'] for s in samples),
            mean_mse_vs_paired_fp16=statistics.fmean(s['mse_vs_paired_fp16'] for s in samples),
            selected_cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in subset),
            any_stage_cv_failed_records=sum(any(s['cv_percent']>=3 for s in r['stage_summaries'].values()) for r in subset),
            per_sample=samples))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--samples',type=int,default=24);p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    p.add_argument('--implementations',type=int,nargs='+',default=[0,4])
    args=p.parse_args()
    if (args.output.exists() or not 1<=args.samples<=24 or args.rounds<1 or args.warmup<0 or args.repeats<2
        or args.inner<2 or 0 not in args.implementations or len(set(args.implementations))!=len(args.implementations)
        or any(i not in (0,1,2,3,4) for i in args.implementations)):
        p.error('fresh output, baseline0, valid unique implementations and positive counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_mixed import validate_fp16_result
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('A100 required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    manifest,mh=inspect_inputs(args.data)
    raw_manifest,rh=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
    raw_index={e['sample_id']:e for e in raw_manifest['samples']}
    args.output.mkdir(parents=True)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as out: out.write(json.dumps(value,allow_nan=False)+'\n')
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),binary_sha256=sha256_file(Path(native.__file__)),
        torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        prepared_manifest_sha256=mh,raw_manifest_sha256=rh,gemm_tune=59,gemm_math_changed=False,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        input_policy='original_fp16_direct_source_quantization',policy='shared GPU; unlocked; no filtering; cyclic order',
        scope='four_mode_conversion_candidate_not_production',timing_contract_version=2))
    rows=[]
    for si,e in enumerate(manifest['samples'][:args.samples]):
        re=raw_index[e['sample_id']]
        if sha256_file(args.data/e['file'])!=e['sha256'] or sha256_file(args.raw_data/re['file'])!=re['sha256']:
            raise ValueError('inputs changed after preflight')
        raw=_load_and_validate_raw(args.raw_data/re['file'],re['layer'],re['projection'])
        x=load_prepared(args.data/e['file'],device='cpu')
        verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']));del x
        for vi,(v,(wf,af)) in enumerate(mf.VARIANTS.items()):
            ws=mf.quantize_source(raw['weight_fp16'].cuda(),wf);acs=mf.quantize_source(raw['activation_fp16'].cuda(),af)
            append('source_provenance.jsonl',dict(sample_id=e['sample_id'],variant=v,weight=source_identity(ws),activation=source_identity(acs)))
            fp=native._benchmark_mixed(mf.PAIRED_BASELINE[v],'compute_only',ws,acs,0,1,2,'64x128x256','row_major')
            validate_fp16_result(fp,ws,acs)
            base=native._benchmark_mixed(v,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,0)
            expected_mse=mse(base['output'],fp['output'])
            for ri in range(args.rounds):
                append('gpu_snapshots.jsonl',dict(sample_id=e['sample_id'],variant=v,round=ri,time=time.time(),
                    gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                for mi,mode in enumerate(('conversion_only','compute_only','cold','steady_state')):
                    for impl in order(args.implementations,si,vi+mi,ri):
                        result=native._benchmark_mixed(v,mode,ws,acs,args.warmup,args.repeats,args.inner,'64x128x256','group_major',59,impl)
                        validate_pair(result,base,mode,impl,args.inner,args.repeats)
                        values={k:list(vs) for k,vs in result['timings_ms'].items()}
                        summaries={k:stats(vs) for k,vs in values.items()}
                        selected='gemm' if mode=='compute_only' else 'total'
                        row=dict(sample_id=e['sample_id'],variant=v,mode=mode,round=ri,implementation=impl,
                            raw_ms=values,stage_summaries=summaries,summary=summaries[selected],selected_stage=selected,
                            total_timing=result['total_timing'],stage_timing_inner_repeats=dict(result['stage_timing_inner_repeats']),
                            weight_cached=result['weight_cached'],activation_prepared=result['activation_prepared'],
                            kernel=dict(result['kernel']),bitwise_equal_current_best=True,mse_vs_current_best=0.,
                            mse_vs_paired_fp16=expected_mse,paired_reference=mf.PAIRED_BASELINE[v])
                        rows.append(row);append('results.jsonl',row)
                        print(e['sample_id'],v,mode,impl,summaries[selected]['median_ms'],flush=True)
            del base,fp,ws,acs,result
        save('summary.json',summarize(rows))
        save('validation.json',dict(passed=True,records=len(rows),samples=si+1,output_bitwise_current_best=True))
    print('FOUR MODE CONVERSION PIPELINE PASSED',flush=True)


if __name__=='__main__': main()

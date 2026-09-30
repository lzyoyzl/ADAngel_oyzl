#!/usr/bin/env python3
"""Paired O3/O7/O8 candidate compute-only measurements on original FP16 trace.

Default isolates the prepared core. --all-modes reuses native conversion and
four-mode timers. Production defaults remain unchanged. No outlier filtering.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats
from roof_reduction_validation import row_scale_layout_reference, compare_output, reference_fp64, mse_regression_ok
from roof_payload_validation import verify_grouped_payload, payload_reorder_bytes_for_stage
from benchmark_a100_mixed import conversion_bytes, integer_reference, validate_fp16_result
from benchmark_a100_mixed_trace import (
    inspect_inputs, inspect_raw_inputs, mse, source_identity, verify_raw_prepared,
)


def measurement_order(tunes, sample_index, variant_index, round_index, mode_index=0):
    """Cyclic balanced order; do not reverse it again and cancel the two-case AB/BA."""
    offset=(sample_index+variant_index+round_index+mode_index)%len(tunes)
    return tunes[offset:]+tunes[:offset]


def summarize(records):
    from adangel.benchmark.metrics import bootstrap_median_ci
    index = {(r['sample_id'], r['variant'], r.get('mode','compute_only'), r['round'], r['tune']): r for r in records}
    if len(index) != len(records):
        raise ValueError('duplicate sample/variant/round/tune')
    result = []
    for variant, mode, tune in sorted({(r['variant'],r.get('mode','compute_only'),r['tune']) for r in records}):
        rows = [r for r in records if (r['variant'],r.get('mode','compute_only'),r['tune']) == (variant,mode,tune)]
        samples = []
        for sid in sorted({r['sample_id'] for r in rows}):
            group = [r for r in rows if r['sample_id'] == sid]
            ratios = [index[(sid, variant, mode, r['round'], -1)]['summary']['median_ms'] /
                      r['summary']['median_ms'] for r in group]
            if len({(r['mse_vs_o0'],r['mse_vs_paired_fp16']) for r in group}) != 1:
                raise ValueError('MSE changed across rounds')
            samples.append(dict(sample_id=sid,
                median_ms=statistics.median(r['summary']['median_ms'] for r in group),
                paired_speedup=statistics.median(ratios), mse_vs_o0=group[0]['mse_vs_o0'],
                mse_vs_paired_fp16=group[0]['mse_vs_paired_fp16']))
        speeds = [r['paired_speedup'] for r in samples]
        median = statistics.median(r['median_ms'] for r in samples)
        result.append(dict(variant=variant, mode=mode, tune=tune, samples=len(samples), records=len(rows),
            median_ms=median, effective_tops=2*4096**3/median/1e9 if mode!='conversion_only' else None,
            paired_speedup_median=statistics.median(speeds),
            paired_speedup_ci95=list(bootstrap_median_ci(speeds,10000,.95,20260930)) if len(samples)>1 else None,
            median_mse_vs_o0=statistics.median(r['mse_vs_o0'] for r in samples),
            median_mse_vs_paired_fp16=statistics.median(r['mse_vs_paired_fp16'] for r in samples),
            cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in rows),
            any_stage_cv_failed_records=sum(any(s['cv_percent']>=3 for s in r.get('stage_summaries',{'selected':r['summary']}).values()) for r in rows),
            per_sample=samples))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data', type=Path, default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config', type=Path, default=Path('configs/trace/llama2_7b_prefill.yaml'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--inner', type=int, default=100)
    p.add_argument('--all-modes', action='store_true')
    p.add_argument('--allow-reassociation',action='store_true',
                   help='Opt in to candidate24-27/34-40 numerical policy; all others remain bitwise gated')
    p.add_argument('--tunes', type=int, nargs='+', default=[-1,1,2,3])
    p.add_argument('--variants', nargs='+', choices=['o3','o7','o8'], default=['o3','o7','o8'])
    args = p.parse_args()
    if (args.output.exists() or not 1<=args.samples<=24 or args.rounds<1 or args.warmup<0 or args.repeats<2 or args.inner<2
        or -1 not in args.tunes or len(set(args.tunes))!=len(args.tunes)
        or any(t not in (-1,0,1,2,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50)+(51,52)+(53,54) for t in args.tunes) or len(set(args.variants))!=len(args.variants)):
        p.error('fresh output, production control (-1), unique valid cases and positive repetitions required')
    if any(t in (43,44) for t in args.tunes) and not args.all_modes:
        p.error('fused conversion candidates43/44 require --all-modes; prepared-core has no source conversion')
    if any(t in (24,25,26,27,34,35,36,37,38,39,40,51,52,53,54) for t in args.tunes) and not args.allow_reassociation:
        p.error('candidates24-27/34-40/51-54 require explicit --allow-reassociation')
    if any(t in (51,52,53,54) for t in args.tunes) and args.variants != ['o3']:
        p.error('row-scale epilogue candidates51/52/53/54 are O3 only')
    if 13 in args.tunes and args.variants != ['o3']:
        p.error('candidate13 is O3 only')
    if any(t in (14,15) for t in args.tunes) and 'o3' in args.variants:
        p.error('asynchronous scale candidates are O7/O8 only')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import load_prepared, sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    if torch.cuda.get_device_capability() != (8,0):
        raise RuntimeError('SM80 required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest, mh = inspect_inputs(args.data)
    raw_manifest, rh = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
    raw_entries = {r['sample_id']:r for r in raw_manifest['samples']}
    args.output.mkdir(parents=True)

    def save(name, obj):
        (args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

    def append(name, obj):
        with (args.output/name).open('a') as stream:
            stream.write(json.dumps(obj,allow_nan=False)+'\n')

    save('environment.json', dict(git_commit=command('git','rev-parse','HEAD'),
        measurement_order_version='cyclic_sample_variant_round_mode_v2',
        binary_sha256=sha256_file(Path(native.__file__)), torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), prepared_manifest_sha256=mh, raw_manifest_sha256=rh,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='real_trace_candidate_four_modes' if args.all_modes else 'real_trace_candidate_prepared_core_only',
        timing='conversion_amortized_end_to_end_direct' if args.all_modes else 'single_execution_cuda_event',
        conversion_timed=args.all_modes,
        policy='unlocked shared GPU; paired sample/round; no filtering',
        input_policy='original_fp16_direct_source_quantization'))
    records = []
    modes=('conversion_only','compute_only','cold','steady_state') if args.all_modes else ('compute_only',)
    for si, entry in enumerate(manifest['samples'][:args.samples]):
        path=args.data/entry['file']; raw_entry=raw_entries[entry['sample_id']]
        raw_path=args.raw_data/raw_entry['file']
        if sha256_file(path)!=entry['sha256'] or sha256_file(raw_path)!=raw_entry['sha256']:
            raise ValueError('input changed after preflight')
        x=load_prepared(path,device='cuda')
        if x.sample_id!=entry['sample_id'] or list(x.shape)!=entry['shape']:
            raise ValueError('sample identity mismatch')
        raw=_load_and_validate_raw(raw_path,raw_entry['layer'],raw_entry['projection'])
        raw_operands=(raw['activation_fp16'],raw['weight_fp16'])
        verify_raw_prepared(x,raw_operands)
        o0=native.benchmark_o0(x.A_int8,x.A_scale,x.W_mxfp4,x.W_scale,'compute_only',0,1,100)['output']
        for vi, variant in enumerate(args.variants):
            if variant=='o3':
                base=native.benchmark('o3','compute_only',x.A_int8,x.A_scale,
                    x.W_mxfp4_g128,x.W_scale_g128,0,1,100,'production')
                values=(base['converted_activation'],x.A_scale,base['converted_weight'],x.W_scale_g128)
                paired=o0; reference_name='o0'
            else:
                wf,af=mf.VARIANTS[variant]
                wsrc=mf.quantize_source(raw_operands[1].cuda(),wf)
                asrc=mf.quantize_source(raw_operands[0].cuda(),af)
                append('source_provenance.jsonl',dict(sample_id=x.sample_id,variant=variant,
                    raw_prepared_replay_bitwise=True,weight=source_identity(wsrc),activation=source_identity(asrc)))
                reference_name=mf.PAIRED_BASELINE[variant]
                fp=native._benchmark_mixed(reference_name,'compute_only',wsrc,asrc,0,1,100,'64x128x256','row_major')
                validate_fp16_result(fp,wsrc,asrc);paired=fp['output']
                base=native._benchmark_mixed(variant,'compute_only',wsrc,asrc,0,1,100,'64x128x256','group_major')
                aq,asc=mf.to_fixed_reference(asrc);wq,wsc=mf.to_fixed_reference(wsrc)
                reference=integer_reference(aq,asc,wq,wsc)
                torch.testing.assert_close(base['output'],reference,rtol=1e-3,atol=1e-3)
                values=(*base['converted_activation'],*base['converted_weight'])
                del reference,aq,asc,wq,wsc
            expected=base['output']
            metrics=dict(mse_vs_o0=mse(expected,o0),mse_vs_paired_fp16=mse(expected,paired))
            semantic=reference_fp64(variant,values) if args.allow_reassociation else None
            for r in range(args.rounds):
                append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=r,time=time.time(),
                    gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                for mode_index, mode in enumerate(modes):
                    order=measurement_order(args.tunes,si,vi,r,mode_index)
                    for tune in order:
                        wall_start=time.time()
                        if not args.all_modes:
                            result=native._benchmark_roof_candidate(variant,tune,*values,args.warmup,args.repeats)
                            timings={'gemm':list(result['gemm_ms'])}
                        else:
                            if variant=='o3':
                                result=native.benchmark('o3',mode,x.A_int8,x.A_scale,x.W_mxfp4_g128,
                                    x.W_scale_g128,args.warmup,args.repeats,args.inner,'production',tune)
                            else:
                                result=native._benchmark_mixed(variant,mode,wsrc,asrc,args.warmup,args.repeats,
                                    args.inner,'64x128x256','group_major',tune)
                            timings={k:list(v) for k,v in result['timings_ms'].items()}
                        wall_end=time.time()
                        y=result['output']
                        payload_checks=verify_grouped_payload(result,tune,values[0],values[2],values[3])
                        if args.allow_reassociation:
                            tree_base=native._benchmark_roof_candidate(variant,36 if tune==37 else 38,*values,0,1)['output'] if tune in (37,39,40) else None
                            numeric=compare_output(y,expected,semantic,tune,dict(result['kernel']),tree_base,
                                row_scale_baseline=row_scale_layout_reference(native,variant,tune,values))
                            actual_metrics=dict(mse_vs_o0=mse(y,o0),mse_vs_paired_fp16=mse(y,paired))
                            numeric['mse_regression_passed']=all(mse_regression_ok(actual_metrics[key],value) for key,value in metrics.items())
                            numeric['baseline_mse']=metrics.copy()
                        else:
                            if y.dtype!=torch.float32 or not torch.isfinite(y).all() or not torch.equal(y.view(torch.int32),expected.view(torch.int32)):
                                raise AssertionError((x.sample_id,variant,tune,mode,'output differs from production'))
                            numeric=dict(bitwise_equal_production=True,mse_vs_production=0.0)
                            actual_metrics=metrics
                        numeric.update(payload_checks)
                        if any(len(t)!=args.repeats for t in timings.values()):
                            raise ValueError('missing timing repetitions')
                        stage='gemm' if mode=='compute_only' else 'total'
                        times=timings[stage]
                        stage_stats={k:stats(v) for k,v in timings.items()}
                        for name, st in stage_stats.items():
                            count=conversion_bytes(variant,name,*x.shape) if 'conversion' in name or mode=='conversion_only' else 0
                            if name=='weight_conversion' or (name=='total' and mode=='conversion_only'):
                                count+=int(result['kernel'].get('weight_scale_reorder_bytes',0))
                            count+=payload_reorder_bytes_for_stage(result['kernel'],mode,name)
                            st['logical_bytes']=count
                            st['logical_gbps']=count/st['median_ms']/1e6 if count else None
                        row=dict(sample_id=x.sample_id,variant=variant,tune=tune,round=r,mode=mode,
                            execution_order=order,order_position=order.index(tune),
                            wall_start_unix=wall_start,wall_end_unix=wall_end,
                            raw_ms=times,summary=stage_stats[stage],stage_timings_ms=timings,stage_summaries=stage_stats,
                            kernel=dict(result['kernel']),conversion_inner_repeats=args.inner if args.all_modes else None,
                            stage_timing_inner_repeats={name:args.inner if 'conversion' in name or mode=='conversion_only' else 1 for name in timings},
                            total_timing='sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event',
                            **numeric,paired_fp16=reference_name,**actual_metrics)
                        records.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,r,'paired round complete',flush=True)
            del values,base,expected,paired
    if len(records)!=args.samples*len(args.variants)*len(args.tunes)*args.rounds*len(modes):
        raise RuntimeError('incomplete measurement coverage')
    save('summary.json',dict(scope='real_trace_four_modes' if args.all_modes else 'real_trace_prepared_core_only',all_24_samples=args.samples==24,
        all_four_modes_completed=args.all_modes,correctness_passed=True,no_filtering=True,
        numerical_policy=('explicit_reassociation_only_for_51_52_53_54' if any(t in (53,54) for t in args.tunes)
            else 'explicit_reassociation_only_for_51_52' if any(t in (51,52) for t in args.tunes)
            else 'explicit_reassociation_only_for_24_25_26_27_34_35_36_37_38_39_40' if any(t in (34,35,36,37,38,39,40) for t in args.tunes)
            else 'explicit_reassociation_only_for_24_25_26_27' if any(t in (26,27) for t in args.tunes)
            else 'explicit_reassociation_only_for_24_25') if args.allow_reassociation else 'bitwise',
        mse_regression_passed=all(r.get('mse_regression_passed',True) for r in records),
        bootstrap_unit='sample; rounds collapsed; descriptive CI (same-trace samples correlated)',
        records=summarize(records)))


if __name__=='__main__':
    main()

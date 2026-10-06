#!/usr/bin/env python3
"""Reproduce v101 diagnostic summaries from preserved raw warp timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from analyze_cta_timeline import analyze

ROOT=Path(__file__).resolve().parents[1]
RUNS={'o3':'o378_roof_v101_o3_timeline_r2','o7':'o378_roof_v101_o7_timeline','o8':'o378_roof_v101_o8_timeline'}


def summarize(root):
    rows=[];files={}
    for variant,folder in RUNS.items():
        base=root/folder;receipt=json.loads((base/'receipt.json').read_text())
        if (receipt['variant']!=variant or receipt['shape']!=[4096]*3
                or receipt['production_default_changed'] or receipt['new_best_result']
                or not receipt['all_instrumented_outputs_bitwise_previous_best']
                or receipt['mse_vs_previous_best']!=0):raise ValueError('diagnostic receipt contract failed')
        files[str((base/'receipt.json').relative_to(root))]=hashlib.sha256((base/'receipt.json').read_bytes()).hexdigest()
        captures=[];quantum=[]
        for i in range(3):
            path=base/f'{variant}_timeline_{i}.npy';raw=np.load(path,allow_pickle=False)
            if raw.shape!=(2048,4,4) or raw.dtype!=np.uint64:raise ValueError('raw timeline geometry failed')
            result=analyze(raw)
            if result!=receipt['captures'][i]:raise ValueError('recalculated timestamp analysis differs')
            files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            timestamps=np.unique(raw[:,:,:2].reshape(-1))
            quantum.append(int(np.gcd.reduce(np.diff(timestamps))))
            captures.append(result)
        median=lambda key:statistics.median(x[key] for x in captures)
        missing=[x['full_capture_missing_capacity_equivalent_ms']/x['globaltimer_span_ms']*100 for x in captures]
        tailmissing=[x['tail_missing_capacity_equivalent_ms']/x['globaltimer_span_ms']*100 for x in captures]
        resources=receipt['resources'];static=receipt['codegen']['liveness']
        symbol=receipt['codegen']['config']['symbol']
        loop=next(x for x in static[symbol]['loops'] if x['kind']=='integer')
        if (not receipt['codegen']['gate']['passed'] or not receipt['codegen']['control_comparison']['passed']
                or any(r['active_blocks_per_SM']!=3 or r['registers_per_thread']!=168 for r in resources.values())
                or any(op.startswith(('LDL','STL')) for op in loop['opcode_counts'])):
            raise ValueError('resource/copy/math diagnostic gate failed')
        rows.append(dict(variant=variant,sample_id=receipt['sample_id'],captures=3,
            median_globaltimer_span_ms=median('globaltimer_span_ms'),
            median_last_CTA_tail_ms=median('after_last_start_ms'),
            three_CTA_SM_time_fraction_percent=statistics.median(x['capacity_time_fraction']['3'] for x in captures)*100,
            full_missing_capacity_equivalent_percent=statistics.median(missing),
            tail_missing_capacity_equivalent_percent=statistics.median(tailmissing),
            observed_timestamp_difference_gcd_ns=quantum,
            all_2048_CTAs_all_108_SMs=True,max_overlap=3,
            resources=resources,hot_local_instructions=0,
            control_pre_median_ms=receipt['control_before']['median_ms'],
            control_post_median_ms=receipt['control_after']['median_ms'],
            control_pre_cv_percent=receipt['control_before']['cv_percent'],
            control_post_cv_percent=receipt['control_after']['cv_percent'],
            instrumented_capture_Event_ms=receipt['instrumented_capture_Event_ms'],
            sampled_output_MSE_vs_paired_FP16=receipt['mse_vs_paired_fp16'],
            bitwise_previous_best=True,output_MSE_vs_previous_best=0.0,
            extension_sha256=receipt['extension_sha256'],profile_source_commit=receipt['source_commit']))
    return dict(scope='v101_read_only_diagnostic_not_a_new_best_version_or_24_sample_performance_test',
        results=rows,artifact_sha256=files,production_default_changed=False,new_speedup_claim=False,
        missing_capacity_not_achievable_speedup_or_rigorous_peak_bound=True,
        clock_unlocked=True,outliers_or_CV_failures_removed=False,
        initial_O3_failure='untimed UInt32 CUDA any unsupported; host-check fix, original failed logs retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):p.error('fresh repository output required')
    result=summarize(a.evidence)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['results'],indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Recompute v118 full24 four-mode paired evidence, without filtering."""
import argparse
import json
from pathlib import Path

from analyze_mx8_warp_lut import analyze as analyze_original_protocol


def analyze(directory):
    # Same 576-row protocol, exact raw Event/statistical/numerical gates.
    result=analyze_original_protocol(directory)
    rows=[json.loads(line) for line in (directory/'results.jsonl').read_text().splitlines()]
    if any(not row['packed_weight_and_square_scratch_checked_after_every_call'] or
           not row['activation_preparation_identical'] or row['modified_stage']!=
           'weight_payload_decode_and_exact_square_sum_only' for row in rows):
        raise ValueError('v118 conversion/weight proof scope drift')
    environment=json.loads((directory/'environment.json').read_text())
    conversion=environment['codegen']['conversion_swar']
    if not conversion['audit']['worth_runtime_validation'] or conversion['GEMM_modified']:
        raise ValueError('candidate compile gate/default drift')
    result['scope']='v118_packed_NVFP4_weight_conversion_identical_v78_GEMM'
    result['conversion_compile_audit']=conversion['audit']
    total=next(r for r in result['stages'] if r['mode']=='conversion_only' and r['stage']=='total')
    cold=next(r for r in result['stages'] if r['mode']=='cold' and r['stage']=='total')
    result['conclusion']=dict(
        conversion_total_gain_confirmed=total['paired_speedup_ci95'][0]>1,
        cold_gain_statistically_positive=cold['paired_speedup_ci95'][0]>1,
        direct_CV_failures_retained=True,GEMM_speedup_claim=False,
        default_unchanged=True,stop_adjacent_Boolean_or_lookup_variants=True,
        effective_GEMM_throughput_goal_not_achieved_by_conversion=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=analyze(args.input)
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""v139 exact paired replay; no new GEMM or promotion claim from timing noise."""
import argparse
import json
from pathlib import Path
from analyze_mx8_warp_lut import analyze as analyze_events


def analyze(directory):
    result = analyze_events(directory)
    env = json.loads((directory/'environment.json').read_text())
    if any(env['args'][k] != v for k, v in dict(samples=24, rounds=3, warmup=1000, repeats=200, inner=100).items()):
        raise ValueError('fixed measurement protocol drift')
    receipt = env['codegen']['conversion_combo']
    if (receipt['GEMM_modified'] or receipt['failed_v126_packed_MX8_candidate_executed'] or
            not all(p['passed'] for p in receipt['retained_entry_comparisons'].values())):
        raise ValueError('retained binary/route evidence failed')
    rows = list(map(json.loads, (directory/'results.jsonl').read_text().splitlines()))
    guards, errors = {}, {}
    for r in rows:
        if (not r['payload_norm_checked_after_every_call'] or not r['weight_preparation_identical'] or
                r['failed_v126_packed_MX8_candidate_executed'] or
                r['modified_stage'] != 'activation_conversion_selection_only'):
            raise ValueError('integration proof scope drift')
        if r['guard']['invalid_ctas'] or guards.setdefault(r['sample_id'],r['guard']) != r['guard']:
            raise ValueError('guard differs between modes/policies or invalid input')
        if errors.setdefault(r['sample_id'],r['mse_vs_paired_fp16']) != r['mse_vs_paired_fp16']:
            raise ValueError('MSE changed between modes/policies')
        wanted = (('roof_o78_nv4_swar_benchmark',1),('roof_o78_mx8_swar_benchmark',0))[r['candidate']]
        if (r['resources']['conversion_host_function'],r['resources']['conversion_selector']) != wanted:
            raise ValueError('wrong actual conversion host route')
    for r in result['stages']:
        r['paired_latency_reduction_percent'] = (1-1/r['paired_speedup'])*100
    selected = {(r['mode'],r['stage']):r for r in result['stages']}
    result.update(scope='v139_existing_O7_conversion_integration_not_new_GEMM',
        conversion_audit=receipt, all_real_integer_path=all(g['fallback_ctas']==0 for g in guards.values()),
        guard_summary=guards, GEMM_speedup_claim=False,
        conclusion=dict(conversion_gain_confirmed=selected['conversion_only','total']['paired_speedup_ci95'][0]>1,
            cold_gain_confirmed=selected['cold','total']['paired_speedup_ci95'][0]>1,
            steady_gain_confirmed=selected['steady_state','total']['paired_speedup_ci95'][0]>1,
            no_new_GEMM_optimization=True, default_unchanged=True, no_filtering=True,
            failed_v126_packed_MX8_candidate_not_reopened=True,
            best_vs_v118_requires_paired_evidence_not_product_of_old_speedups=True))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.input)
    with a.output.open('x') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('conversion_audit','guard_summary')},indent=2))

#!/usr/bin/env python3
"""Freeze v137 raw receipts and independently replay full24 paired summaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from benchmark_o3_eight_chain_probe import summarize
from benchmark_a100_o1 import stats

RUN='runs/o378_roof_v137_full24'
ALLOWED=('.json','.jsonl','.txt','.log','.sass','.ptx','.cu','.cuh','.cpp')
PREFIXES=('reports/o378_roof_v137_', 'runs/o378_roof_v137_')
EXPECTED_EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory):
    run=directory/RUN
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    env=json.loads((run/'environment.json').read_text())
    validation=json.loads((run/'validation.json').read_text())
    resources=json.loads((run/'resources.json').read_text())
    provenance=json.loads((run/'input_provenance.json').read_text())
    receipt=json.loads((directory/'reports/o378_roof_v137_o3_route_cohort_fixed_codegen/codegen.json').read_text())
    ids=[e['sample_id'] for e in provenance['samples']]
    if len(ids)!=24 or len(set(ids))!=24 or len(rows)!=144:raise ValueError('complete full24x3 pair required')
    args=env['args']
    if any(args[k]!=v for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items()):
        raise ValueError('measurement contract drift')
    if args['modes']!=['compute_only'] or env['extension_sha256']!=EXPECTED_EXTENSION:
        raise ValueError('scope/extension drift')
    if not validation['passed'] or not receipt['cost_gate']['passed']:raise ValueError('gate/validation failed')
    rebuilt=summarize(rows,ids,3,['compute_only'])
    if rebuilt!=json.loads((run/'summary.json').read_text())['records']:raise ValueError('summary replay mismatch')
    for row in rows:
        if row['mse_vs_current_best']!=0 or not row['payload_bitwise'] or not row['bitwise_equal_current_best']:
            raise ValueError('exact regression')
        if not row['kernel']['two_native_int4'] or row['kernel']['int8_tensor_core']:raise ValueError('ISA drift')
        if row['stage_timing_inner_repeats']!={'gemm':1,'total':1}:raise ValueError('timing not direct')
        for stage,times in row['raw_ms'].items():
            if len(times)!=200 or not all(math.isfinite(t) and t>0 for t in times):raise ValueError('bad raw durations')
            # CPU NumPy reductions may differ in the last binary digits across
            # the A100 environment and local evidence reader; no GPU sample is edited.
            expected=stats(times);saved=row['stage_summaries'][stage]
            if expected.keys()!=saved.keys() or any(not math.isclose(expected[k],saved[k],
                    rel_tol=1e-12,abs_tol=1e-12) for k in expected):
                raise ValueError('stage statistics mismatch')
    safety={}
    for tool in ('synccheck','memcheck','racecheck'):
        log=(directory/f'reports/o378_v137_{tool}.log').read_text()
        expected='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors'
        proof=json.loads((directory/f'runs/o378_roof_v137_{tool}/validation.json').read_text())
        if expected not in log or not proof['passed'] or len(proof['checks'])!=4:raise ValueError('safety incomplete')
        safety[tool]=dict(passed=True,scope=proof['scope'],cases=len(proof['checks']))
    pair={r['implementation']:r for r in rebuilt};base,new=pair[0],pair[1]
    per_round=[]
    for ri in range(3):
        p={(r['sample_id'],r['implementation']):r for r in rows if r['round']==ri}
        ratios=[p[(sid,0)]['summary']['median_ms']/p[(sid,1)]['summary']['median_ms'] for sid in ids]
        per_round.append(dict(round=ri,paired_throughput_change_percent=(statistics.median(ratios)-1)*100))
    return dict(scope='v137 two-native-INT4 route cohorts versus O3 v89; full24x3 direct Event',
        records=144,samples=24,rounds=3,source_commit=env['git_commit'],
        candidate_cubin_sha256=receipt['cubin_sha256'],control_cubin_sha256=receipt['baseline_cubin_sha256'],
        control_median_ms=base['median_ms'],candidate_median_ms=new['median_ms'],
        paired_speedup=new['paired_speedup'],paired_speedup_ci95=new['paired_speedup_ci95'],
        paired_throughput_change_percent=(new['paired_speedup']-1)*100,
        paired_latency_change_percent=(1/new['paired_speedup']-1)*100,per_round=per_round,
        control_cv_failed=base['selected_cv_failed_records'],candidate_cv_failed=new['selected_cv_failed_records'],
        records_per_side=72,bitwise_best_all=True,mse_vs_best=0.0,
        median_mse_vs_o0=new['median_mse_vs_paired_fp16'],mean_mse_vs_o0=new['mean_mse_vs_paired_fp16'],
        all_real_integer_path=all(r['guard_status']==0 for r in rows),
        numerical_validation_cases=len(validation['checks']),rejected_invalid_cases=len(validation['rejected']),
        extra_grouped_cases=len(validation['grouped_coordinates_checks']),safety=safety,
        resources={k:resources[k] for k in ('v89','v137')},
        integer_loop_local_loads=receipt['cost_gate']['hot_local_loads'],
        integer_loop_local_stores=receipt['cost_gate']['hot_local_stores'],
        fallback_loop=(next(r for r in receipt['liveness']['loops'] if r['kind']=='fp32_fallback')),
        formal_extension_sha256=EXPECTED_EXTENSION,production_default_changed=False,
        four_mode_expansion_justified=new['paired_speedup_ci95'][0]>1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-existing',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() and not a.replay_existing) or not out.is_relative_to(ROOT):
        p.error('fresh repository output, or explicit byte-verified replay required')
    if digest(a.archive)!=a.sha256:raise ValueError('archive SHA mismatch')
    records=[]
    with tarfile.open(a.archive,'r:gz') as archive:
        members=archive.getmembers();names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or member.name in names or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe archive member')
            names.add(member.name)
            if not (member.name.startswith(PREFIXES) or member.name.startswith('reports/o378_v137_')):
                raise ValueError('out of scope member: '+member.name)
        out.mkdir(parents=True,exist_ok=a.replay_existing)
        for member in members:
            if not member.isfile() or PurePosixPath(member.name).suffix not in ALLOWED:continue
            data=archive.extractfile(member).read();target=out/member.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.read_bytes()!=data:raise ValueError('existing raw evidence differs: '+member.name)
            else:target.write_bytes(data)
            records.append(dict(path=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    result=analyze(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'index.json').write_text(json.dumps(dict(archive_sha256=a.sha256,files=sorted(records,key=lambda r:r['path']),
        analysis_sha256=digest(out/'analysis.json'),artifact_count=len(records),artifact_bytes=sum(r['bytes'] for r in records),
        binary_artifacts_in_archive_not_git=True,source_commit=result['source_commit']),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fallback_loop'},indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Freeze/recompute v135 first full24 runtime of the unchanged v115 cubin."""
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

RUN='runs/o378_roof_v135_full24_checked'
PREFIXES=('runs/o378_roof_v135_full24/','runs/o378_roof_v135_full24_contract/',RUN+'/')
LOGS=('reports/o378_v135_runtime.log','reports/o378_v135_contract_runtime.log',
      'reports/o378_v135_checked_runtime.log','reports/o378_v135_final_checks.log')
ALLOWED=('.json','.jsonl','.txt','.log','.sass','.ptx','.cu','.cuh','.cpp')
EXPECTED_EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(directory):
    run=directory/RUN
    rows=[json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    env=json.loads((run/'environment.json').read_text())
    validation=json.loads((run/'validation.json').read_text())
    resources=json.loads((run/'resources.json').read_text())
    provenance=json.loads((run/'input_provenance.json').read_text())
    ids=[e['sample_id'] for e in provenance['samples']]
    if len(ids)!=24 or len(set(ids))!=24 or len(rows)!=144:raise ValueError('complete full24x3 pair required')
    args=env['args']
    if any(args[k]!=v for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items()):
        raise ValueError('measurement contract drift')
    if args['modes']!=['compute_only'] or env['extension_sha256']!=EXPECTED_EXTENSION:
        raise ValueError('run scope or formal extension drift')
    if not validation['passed'] or len(validation['dp2a_extra'])!=4:raise ValueError('validation incomplete')
    rebuilt=summarize(rows,ids,3,['compute_only'])
    saved=json.loads((run/'summary.json').read_text())
    if rebuilt!=saved['records']:raise ValueError('summary replay mismatch')
    for row in rows:
        if row['mse_vs_current_best']!=0 or not row['payload_bitwise'] or not row['bitwise_equal_current_best']:
            raise ValueError('exact regression')
        if not row['kernel']['two_native_int4'] or row['kernel']['int8_tensor_core']:
            raise ValueError('matrix ISA contract drift')
        if row['stage_timing_inner_repeats']!={'gemm':1,'total':1}:raise ValueError('timing not direct')
        for times in row['raw_ms'].values():
            if len(times)!=200 or not all(math.isfinite(t) and t>0 for t in times):raise ValueError('bad raw samples')
    pair={r['implementation']:r for r in rebuilt}
    base,new=pair[0],pair[1]
    coverage={sid:[r['kernel']['dp2a_cta_fraction'] for r in rows if r['sample_id']==sid and r['implementation']==1] for sid in ids}
    if any(len(set(v))!=1 for v in coverage.values()):raise ValueError('coverage changes across rounds')
    per_round=[]
    for ri in range(3):
        p={(r['sample_id'],r['implementation']):r for r in rows if r['round']==ri}
        ratios=[p[(sid,0)]['summary']['median_ms']/p[(sid,1)]['summary']['median_ms'] for sid in ids]
        per_round.append(dict(round=ri,paired_throughput_change_percent=(statistics.median(ratios)-1)*100))
    return dict(scope='v135 frozen v115 first complete GPU measurement; not new CUDA optimization',
        records=144,samples=24,rounds=3,raw_cuda_event_durations=sum(len(t) for r in rows for t in r['raw_ms'].values()),
        source_commit=env['git_commit'],control='O3 v89',candidate='frozen O3 v115 DP2A',
        control_median_ms=base['median_ms'],candidate_median_ms=new['median_ms'],
        paired_speedup=new['paired_speedup'],paired_speedup_ci95=new['paired_speedup_ci95'],
        paired_throughput_change_percent=(new['paired_speedup']-1)*100,
        paired_latency_change_percent=(1/new['paired_speedup']-1)*100,
        per_round=per_round,control_cv_failed=base['selected_cv_failed_records'],
        candidate_cv_failed=new['selected_cv_failed_records'],records_per_side=72,
        bitwise_best_all=True,mse_vs_best=0.0,
        median_mse_vs_o0=new['median_mse_vs_paired_fp16'],mean_mse_vs_o0=new['mean_mse_vs_paired_fp16'],
        mean_dp2a_cta_fraction=statistics.fmean(v[0] for v in coverage.values()),
        per_sample_dp2a_cta_fraction={k:v[0] for k,v in coverage.items()},
        resources={k:resources[k] for k in ('v89','frozen_v115')},
        numerical_validation_cases=len(validation['checks']),
        rejected_invalid_cases=len(validation['rejected']),
        extra_grouped_cases=len(validation['grouped_coordinates_checks']),
        extra_three_path_cases=len(validation['dp2a_extra']),
        historical_compile_gate_still_failed=True,formal_extension_sha256=EXPECTED_EXTENSION,
        production_default_changed=False,new_NCU=False,new_sanitizer=False,
        four_mode_expansion_justified=new['paired_speedup_ci95'][0]>1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    if digest(a.archive)!=a.sha256:raise ValueError('archive SHA mismatch')
    records=[]
    with tarfile.open(a.archive,'r:gz') as archive:
        members=archive.getmembers();names=set()
        for member in members:
            path=PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or member.name in names or
                not (member.isfile() or member.isdir())):raise ValueError('unsafe archive member')
            names.add(member.name)
            if not (any(member.name.startswith(x) or member.name==x.rstrip('/') for x in PREFIXES) or member.name in LOGS):
                raise ValueError('out of scope archive member: '+member.name)
        out.mkdir(parents=True)
        for member in members:
            if not member.isfile() or PurePosixPath(member.name).suffix not in ALLOWED:continue
            data=archive.extractfile(member).read();target=out/member.name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records.append(dict(path=member.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    result=analyze(out)
    (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    index=dict(archive_sha256=a.sha256,files=sorted(records,key=lambda r:r['path']),
        analysis_sha256=digest(out/'analysis.json'),artifact_count=len(records),
        artifact_bytes=sum(r['bytes'] for r in records),binary_artifacts_in_archive_not_git=True,
        source_commit=result['source_commit'],original_v115_evidence_unchanged=True)
    (out/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('per_sample')},indent=2))


if __name__=='__main__':main()

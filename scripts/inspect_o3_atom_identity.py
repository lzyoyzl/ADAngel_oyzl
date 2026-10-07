#!/usr/bin/env python3
"""v116: bound real-data, per-G128 N8 identity-scale fusion opportunities.

Not v77's all-unit diagnostic or v108's K-constant sorted whole tiles.
Do not develop a new GEMM if even free classification/rematerialization
would remove <5% of the current loop work. No GPU/timing or quantization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OBS=Path('docs/evidence/a100_o378_roof_v108/reports/o378_roof_v108_profile_buckets')
AOBS=Path('docs/evidence/a100_o378_roof_v105/reports/o378_roof_v105_runtime_factor_r2')
EXPECTED={f'layer_{i:02d}_{p}' for i in (0,6,12,18,24,31) for p in ('q_proj','k_proj','v_proj','o_proj')}
LOOP=323
REMOVABLE=64
THRESHOLD=5.0
SOURCES=('scripts/inspect_o3_atom_identity.py','tests/cuda/validate_o3_atom_columns.cu',
         'csrc/sm80/o3_row_scale_epilogue_candidate.cuh','csrc/sm80/o78_unsigned_payload_candidate.cuh',
         'scripts/inspect_scale_profile_buckets.py')


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def atom_statistics(codes):
    c=np.asarray(codes)
    if c.dtype!=np.uint8 or c.ndim!=2 or c.shape[1]!=32 or not c.shape[0] or c.shape[0]%128:
        raise ValueError('natural uint8[N128,32] scale required')
    if np.any((c==0)|(c==255)):raise ValueError('normal UE8M0 codes 1..254 required')
    # The original exact factor is 2**(code - min_code_per_column).
    # Each N8 instruction group has to be unit, not just some lanes/outputs.
    identity=c==c.min(axis=1)[:,None]
    panels=identity.reshape(c.shape[0]//8,8,32).all(axis=1)
    fraction=float(panels.mean())
    return dict(columns=c.shape[0],groups=32,native_atom_columns=8,
        native_N8_group_panels=int(panels.size),identity_panels=int(panels.sum()),
        identity_panel_fraction=fraction,per_group_identity_panels=panels.sum(axis=0).tolist(),
        identity_panels_per_N128_tile=panels.reshape(-1,16,32).sum(axis=(1,2)).tolist(),
        current_integer_loop_static_instructions=LOOP,max_eliminable_updates_per_loop=REMOVABLE,
        optimistic_instruction_work_reduction_percent=fraction*REMOVABLE/LOOP*100,
        optimistic_mean_eliminable_updates=REMOVABLE*fraction,
        exclusions='classification, dispatch, flag loads, scheduling/register cost and all fullK fallback ignored',
        interpretation='optimistic instruction work budget, not latency/speedup/kernel peak')


def observe(root=ROOT):
    base=root/OBS
    original=[json.loads(s) for s in (base/'results.jsonl').read_text().splitlines()]
    selected=[r for r in original if r['variant']=='o3']
    if len(selected)!=24 or {r['sample_id'] for r in selected}!=EXPECTED:
        raise ValueError('all24 original O3 samples required')
    env=json.loads((base/'environment.json').read_text())
    historical=subprocess.check_output(['git','show',env['git_commit']+':scripts/inspect_scale_profile_buckets.py'],cwd=root)
    if hashlib.sha256(historical).hexdigest()!=env['script_sha256']:
        raise ValueError('original snapshot collector source SHA drift')
    rows=[]
    for r in selected:
        p=base/r['artifact']
        if digest(p)!=r['artifact_sha256']:raise ValueError('snapshot artifact SHA drift')
        with np.load(p,allow_pickle=False) as data:c=data['scale_codes']
        if c.shape!=(4096,32) or hashlib.sha256(c.tobytes()).hexdigest()!=r['scale_codes_sha256']:
            raise ValueError('full4096 scale array SHA/shape drift')
        rows.append(dict(sample_id=r['sample_id'],variant='o3',
            artifact=str((OBS/r['artifact']).as_posix()),artifact_sha256=r['artifact_sha256'],
            scale_codes_sha256=r['scale_codes_sha256'],prepared_sha256=r['prepared_sha256'],
            statistics=atom_statistics(c)))
    return rows


def summarize(rows):
    if len(rows)!=24 or {r['sample_id'] for r in rows}!=EXPECTED:
        raise ValueError('complete unique24 observations required')
    work=statistics.mean(r['statistics']['optimistic_instruction_work_reduction_percent'] for r in rows)
    return dict(scope='O3_per_G128_native_N8_identity_work_budget_not_GPU_performance',samples=24,
        mean_identity_panel_fraction=statistics.mean(r['statistics']['identity_panel_fraction'] for r in rows),
        optimistic_loop_instruction_work_reduction_percent=work,opportunity_threshold_percent=THRESHOLD,
        opportunity_gate=work>=THRESHOLD,
        decision='compile_one_runtime_identity_candidate_if_gate_passes_else_stop_no_new_GEMM',
        source_quantization_unchanged=True,ordering_unchanged=True,no_filtering=True,
        candidate_GEMM_implemented=False,candidate_GPU_launched=False,new_performance_result=False,
        new_MSE_result=False,production_default_changed=False,
        exclusions='all classification/dispatch/register/flag preparation and fallback costs ignored')


def activation_statistics(factors,status):
    f=np.asarray(factors);st=np.asarray(status)
    if f.dtype!=np.int32 or f.ndim!=2 or f.shape[0]!=32 or not f.shape[1] or f.shape[1]%64 or np.any(f<0):
        raise ValueError('nonnegative INT32 factors[32,M64] required')
    if st.dtype!=np.uint32 or st.shape!=(f.shape[1],) or np.any(st>2):
        raise ValueError('original UINT32 row-status required')
    # For nonnegative INTEGER factors, Af*Wf==1 entails Af==Wf==1.
    # Unlike DP2A's <=15 condition, Wf==0 cannot invalidate this bound.
    valid=(f==1)&(st[None,:]==0)
    panels=valid.reshape(32,f.shape[1]//16,16).all(axis=2)
    fraction=float(panels.mean())
    return dict(rows=f.shape[1],groups=32,native_atom_rows=16,
        native_M16_group_panels=int(panels.size),necessary_A_unit_panels=int(panels.sum()),
        necessary_A_unit_fraction_upper_bound=fraction,
        current_integer_loop_static_instructions=383,max_eliminable_coefficient_and_update_ops=128,
        optimistic_instruction_work_reduction_percent=fraction*128/383*100,
        bound_kind='rigorous A-only necessary-condition upper bound; not complete A/W eligibility',
        proof='nonnegative integer Af*Wf==1 implies Af==Wf==1, including Wf=0 exclusion',
        exclusions='W unit condition, CTA fallback, classification/dispatch/flag/register costs ignored')


def observe_activation(root=ROOT):
    from inspect_scale_profile_buckets import observation_authority
    _,old=observation_authority(root,EXPECTED,digest)
    rows=[]
    for (sid,variant),r in sorted(old.items()):
        with np.load(root/AOBS/r['artifact'],allow_pickle=False) as data:
            f=data['factors'];st=data['row_status']
        if hashlib.sha256(f.tobytes()).hexdigest()!=r['factors_sha256']:
            raise ValueError('original factor array SHA drift')
        rows.append(dict(sample_id=sid,variant=variant,artifact=str((AOBS/r['artifact']).as_posix()),
            artifact_sha256=r['artifact_sha256'],factors_sha256=r['factors_sha256'],
            raw_sha256=r['raw_sha256'],source=r['source'],statistics=activation_statistics(f,st)))
    return rows


def summarize_activation(rows):
    expected={(sid,v) for sid in EXPECTED for v in ('o7','o8')}
    if len(rows)!=48 or {(r['sample_id'],r['variant']) for r in rows}!=expected:
        raise ValueError('all24 O7/O8 necessary-condition observations required')
    result=[]
    for variant in ('o7','o8'):
        values=[r['statistics'] for r in rows if r['variant']==variant]
        work=statistics.mean(r['optimistic_instruction_work_reduction_percent'] for r in values)
        result.append(dict(variant=variant,samples=24,
            mean_A_only_unit_fraction_upper_bound=statistics.mean(r['necessary_A_unit_fraction_upper_bound'] for r in values),
            optimistic_loop_instruction_work_reduction_percent_upper_bound=work,
            opportunity_gate=work>=THRESHOLD,opportunity_threshold_percent=THRESHOLD,
            bound_kind='A-only necessary condition, not measured W matching or final dispatch coverage'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    sha=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'V12.8.93' not in version or sha!='db1c288993354c88e551c40c19a8fb93a774a241':
        raise ValueError('pinned toolchain required')
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one formal extension required')
    ext=digest(extensions[0]);out.mkdir(parents=True)
    cmd=[str(cuda/'nvcc'),'-O2','-std=c++17','-arch=sm_80','--expt-relaxed-constexpr',
        '-I'+str(cutlass/'include'),str(ROOT/SOURCES[1]),'-o',str(out/'validate_coordinates')]
    with (out/'coordinate_build.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'TMPDIR':str(ROOT/'tmp')})
    mapping=json.loads(subprocess.check_output([str(out/'validate_coordinates')],text=True))
    if not mapping['passed'] or mapping['gpu_execution'] or mapping['columns_per_native_atom']!=8:
        raise ValueError('host-only exact CuTe N8 mapping verification failed')
    rows=observe();activation_rows=observe_activation();summary=summarize(rows)
    summary['O7_O8_necessary_condition_bounds']=summarize_activation(activation_rows)
    if digest(extensions[0])!=ext:raise ValueError('formal extension changed')
    env=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={s:digest(ROOT/s) for s in SOURCES},nvcc=version,cutlass_commit=sha,
        commands=[cmd,[str(out/'validate_coordinates')]],
        snapshots_environment_sha256=digest(ROOT/OBS/'environment.json'),
        snapshots_results_sha256=digest(ROOT/OBS/'results.jsonl'),
        activation_snapshots_environment_sha256=digest(ROOT/AOBS/'environment.json'),
        activation_snapshots_results_sha256=digest(ROOT/AOBS/'results.jsonl'),
        formal_extension_sha256_before=ext,formal_extension_sha256_after=ext,
        no_GPU_kernel_launched=True,no_new_quantization=True)
    for name,value in (('environment.json',env),('coordinates.json',mapping),('summary.json',summary)):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (out/'results.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows+activation_rows))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""v115: exact S16/U8 dot fusion feasibility, not a GEMM performance screen.

Preserve two native INT4 matrices, G128 and source quantization. First reuse
SHA-checked full24 factors, then inspect CUDA12.8/SM80 scalar-dot lowering.
Stop before any candidate GEMM if DP2A expands to multiple arithmetic ops.
This is not v61/v67's scalar low+16*high recompose, v72 coefficient ordering,
v78 merged MMA, v87 coefficient shifting or v98/v104 warp geometry.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
O3=Path('docs/evidence/a100_o378_roof_v108/reports/o378_roof_v108_profile_buckets')
O78=Path('docs/evidence/a100_o378_roof_v105/reports/o378_roof_v105_runtime_factor_r2')
SOURCE='tests/cuda/probe_dp2a_recompose.cu'
SYMBOL='adangel_dp2a_recompose_cost'
CONTROL='adangel_scalar_recompose_cost'
FACTOR_LIMIT=15
LOW_BOUND=128*15*8
HIGH_BOUND=128*8*8
PTX_REFERENCE='https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(low,high,factor,accumulator=0):
    if not (-LOW_BOUND<=low<=LOW_BOUND and -HIGH_BOUND<=high<=HIGH_BOUND):
        raise ValueError('proved G128 integer dot ranges required')
    if not isinstance(factor,int) or not 0<=factor<=FACTOR_LIMIT:
        raise ValueError('packed [factor,16*factor] must fit U8')
    packed=(low&65535)|((high&65535)<<16)
    coefficients=factor*4097
    signed16=lambda n: (n&32767)-(n&32768)
    result=accumulator+signed16(packed)*(coefficients&255)+signed16(packed>>16)*((coefficients>>8)&255)
    if result!=accumulator+(low+16*high)*factor:
        raise AssertionError('not an exact integer dot recompose')
    if not -(1<<31)<=result<(1<<31):
        raise ValueError('existing full-K INT32 prefix guard still mandatory')
    return result


def data_gate(root=ROOT):
    """O3 scale gate; O7/O8 conditional A-only bound for positive W factors."""
    rows=[]
    originals=[json.loads(s) for s in (root/O3/'results.jsonl').read_text().splitlines()]
    observations=[json.loads(s) for s in (root/O78/'results.jsonl').read_text().splitlines()]
    expected={r['sample_id'] for r in originals if r['variant']=='o3'}
    if len(expected)!=24 or len(observations)!=48 or {(r['sample_id'],r['variant']) for r in observations}!={(s,v) for s in expected for v in ('o7','o8')}:
        raise ValueError('all24 original-source observations required')
    for r in originals:
        if r['variant']!='o3': continue
        p=root/O3/r['artifact']
        if digest(p)!=r['artifact_sha256']:raise ValueError('O3 snapshot SHA drift')
        with np.load(p,allow_pickle=False) as arrays:c=arrays['scale_codes']
        if c.dtype!=np.uint8 or c.shape!=(4096,32) or digest_bytes(c)!=r['scale_codes_sha256']:
            raise ValueError('O3 scale array contract drift')
        codes=c.astype(np.int16)
        good=(codes.min(axis=1)>=1)&(codes.max(axis=1)<255)&(codes.max(axis=1)-codes.min(axis=1)<=3)
        tiles=good.reshape(32,128).all(axis=1)
        rows.append(dict(sample_id=r['sample_id'],variant='o3',artifact=str((O3/r['artifact']).as_posix()),
            artifact_sha256=r['artifact_sha256'],eligible_n_tiles=int(tiles.sum()),total_n_tiles=32,
            eligible_fraction=float(tiles.mean()),condition='normal UE8M0 and max_code-min_code<=3',
            proof='exact factor<=8; low/high dots fit S16; original prefix/base/input guards retained',
            bound_kind='scale-range eligibility, not a launch/correctness/performance result'))
    for r in observations:
        p=root/O78/r['artifact']
        if not r['v99_source_exact'] or digest(p)!=r['artifact_sha256']:raise ValueError('original O7/O8 factor authority drift')
        with np.load(p,allow_pickle=False) as arrays:f=arrays['factors'];status=arrays['row_status']
        if f.dtype!=np.int32 or f.shape!=(32,4096) or np.any(f<0) or digest_bytes(f)!=r['factors_sha256']:
            raise ValueError('O7/O8 factor contract drift')
        if status.dtype!=np.uint32 or status.shape!=(4096,):raise ValueError('row guard contract drift')
        good=(f<=15).all(axis=0)&(status==0)
        tiles=good.reshape(64,64).all(axis=1)
        rows.append(dict(sample_id=r['sample_id'],variant=r['variant'],artifact=str((O78/r['artifact']).as_posix()),
            artifact_sha256=r['artifact_sha256'],eligible_m_tiles_upper_bound=int(tiles.sum()),total_m_tiles=64,
            eligible_fraction_upper_bound=float(tiles.mean()),
            condition='A_factor<=15 for every G128; assumes W_factor>=1; W factors not inspected',
            bound_kind='conditional A-only upper bound; zero W panels can invalidate this bound'))
    summary=[]
    for v in ('o3','o7','o8'):
        selected=[r for r in rows if r['variant']==v]
        field='eligible_fraction' if v=='o3' else 'eligible_fraction_upper_bound'
        coverage=statistics.mean(r[field] for r in selected)
        summary.append(dict(variant=v,samples=len(selected),mean_coverage=coverage,
            coverage_kind='scale-range fraction' if v=='o3' else 'conditional A-only upper bound (W_factor>=1)',
            minimum_coverage=0.5,data_gate=coverage>=0.5))
    return rows,summary


def digest_bytes(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def entries(sass):
    result={}
    for block in re.split(r'(?=\s*Function\s*:\s*)',sass):
        m=re.match(r'\s*Function\s*:\s*(\w+)',block)
        if m is None or m[1] not in (SYMBOL,CONTROL):continue
        ins=[]
        for line in block.splitlines():
            x=re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*;',line)
            if x:ins.append(dict(pc=hex(int(x[1],16)),instruction=re.sub(r'^@!?\w+\s+','',x[2])))
        result[m[1]]=dict(instructions=ins,opcode_counts=dict(Counter(x['instruction'].split()[0] for x in ins)))
    if set(result)!={SYMBOL,CONTROL}:raise ValueError('both exact cost-probe entries required')
    return result


def isa_gate(entry):
    """Predeclared: one native S16/U8 DP2A, not two IDP4A plus extra shifts."""
    counts=entry['opcode_counts']
    native=sum(v for k,v in counts.items() if re.fullmatch(r'IDP\.2A(?:\.LO)?\.S16\.U8(?:\..*)?',k))
    dots=sum(v for k,v in counts.items() if k.startswith('IDP.'))
    return dict(native_s16_u8_dp2a=native==1,exactly_one_dot_instruction=dots==1,
        native_dot_count=native,all_dot_count=dots,
        passed=native==1 and dots==1,
        scope='scalar ISA gate only; still need complete MMA/guard/resource/24sample/MSE/timing acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project-only output required')
    rows,data=data_gate()
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'V12.8.93' not in version:raise ValueError('pinned CUDA12.8.93 required')
    out.mkdir(parents=True)
    r=dict(scope='v115_scalar_DP2A_lowering_and_full24_scale_range_gate',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={s:digest(ROOT/s) for s in (SOURCE,'scripts/probe_dp2a_recompose.py')},
        references=[PTX_REFERENCE],data_summary=data,commands=[],nvcc=version,
        low_dot_abs_bound=LOW_BOUND,high_dot_abs_bound=HIGH_BOUND,coefficient_limit=FACTOR_LIMIT,
        source_quantization_unchanged=True,production_default_changed=False,
        candidate_GEMM_implemented=False,candidate_GPU_launched=False,new_performance_result=False,new_MSE_result=False)
    extension=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extension)!=1:raise ValueError('one existing formal extension required')
    r['formal_extension_sha256_before']=digest(extension[0])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','-arch=sm_80','-lineinfo',str(ROOT/SOURCE)]
    cubin=out/'dp2a_recompose.cubin'
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/'dp2a_recompose.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'dp2a_recompose.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    e=entries((out/'dp2a_recompose.sass').read_text());gate=isa_gate(e[SYMBOL])
    r.update(entries=e,isa_gate=gate,
        formal_extension_sha256_after=digest(extension[0]),
        decision='complete_one_O3_GEMM_candidate_if_both_gates_pass_else_stop',
        proceed_with_O3_GEMM=gate['passed'] and data[0]['data_gate'])
    if r['formal_extension_sha256_after']!=r['formal_extension_sha256_before']:raise ValueError('formal extension changed')
    (out/'data_rows.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in rows))
    r['artifact_sha256']={f.name:digest(f) for f in out.iterdir() if f.is_file()}
    (out/'analysis.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(data=data,isa_gate=gate,proceed=r['proceed_with_O3_GEMM'],
        candidate_GPU_launched=False,new_performance_result=False),indent=2),flush=True)


if __name__=='__main__':main()

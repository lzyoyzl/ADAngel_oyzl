#!/usr/bin/env python3
"""Reuse v110 cubin; run only the TWO individually passed, unscaled shells.

Do not relax the rejected scaled-shell gate or rebuild its kernel. No real
trace/MSE/default change. Old compilation receipt and repairs remain frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from probe_o78_shell_capacity import ROOT,SYMBOLS,STEM,compile_gate,hot_loop,SHARED


def verify_compilation(directory):
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    r=json.loads((directory/'analysis.json').read_text())
    if r['source_commit']!='e86e28b3a63b06357b423239cbd5e82ee77a6f94':
        raise ValueError('frozen r3 compilation required')
    for path,digest in r['sources'].items():
        content=subprocess.check_output(['git','show',r['source_commit']+':'+path],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('original source SHA mismatch')
        if path not in ('csrc/sm80/roof_o78_shell_capacity_driver.cpp',) and sha(ROOT/path)!=digest:
            raise ValueError('CUDA/source changed since compilation: '+path)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('compile artifact SHA mismatch: '+name)
    rows={s:hot_loop((directory/'liveness.txt').read_text(),s) for s in SYMBOLS}
    if rows!=r['liveness'] or compile_gate(rows)!=r['compile_gate']:
        raise ValueError('original static evidence changed')
    checks=r['compile_gate']['checks']
    if not all(v for check in checks[:2] for k,v in check.items() if k!='symbol'):
        raise ValueError('unscaled shell gate failed')
    if checks[2]['no_hot_local'] or r.get('runtime'):
        raise ValueError('expected rejected scaled shell and no old GPU run')
    if not r['control_comparison']['passed'] or r['production_default_changed']:
        raise ValueError('control/default changed')
    return r


def summarize(rows):
    if len(rows)!=6 or {(r['round'],r['mode']) for r in rows}!={(r,m) for r in range(3) for m in (0,1)}:
        raise ValueError('three rounds of BOTH selected shells required')
    statistics=[]
    for row in rows:
        if (row['symbol']!=SYMBOLS[row['mode']] or row['grid']!=[32,64] or row['groups']!=256
                or row['threads']!=128 or row['active_ctas_per_sm']!=3
                or row['validation_checks']!=12 or not row['checksum_passed']
                or row['shared_reserved_bytes']!=SHARED):
            raise ValueError('runtime/checksum/scope mismatch')
        x=np.asarray(row['raw_ms'],dtype=np.float64)
        if x.shape!=(200,) or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('bad raw Event data')
        statistics.append(dict(mode=row['mode'],round=row['round'],median_ms=float(np.median(x)),
            mean_ms=float(np.mean(x)),normalized32_groups_ms=float(np.median(x)/8),
            cv_percent=float(np.std(x)/np.mean(x)*100),p5_ms=float(np.quantile(x,.05)),p95_ms=float(np.quantile(x,.95))))
    return dict(records=6,statistics=statistics,
        modes=[dict(mode=mode,normalized32_groups_ms=float(np.median([
            x['normalized32_groups_ms'] for x in statistics if x['mode']==mode]))) for mode in (0,1)],
        original_experiment_MSE_measured=False,real_trace_latency_measured=False,
        normalized_times_are_diagnostic_not_kernel_peak=True,no_filtering=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compilation',type=Path,default=Path('reports/o378_roof_v110_shell_capacity_r3'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();directory=args.compilation.resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not directory.is_relative_to(ROOT):
        p.error('existing repository compile directory and fresh output required')
    prior=verify_compilation(directory);sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1 or sha(extensions[0])!=prior['extension_sha256_after']:
        raise ValueError('production extension drift')
    out.mkdir(parents=True);cuda=Path('/usr/local/cuda-12.8/bin')
    r=dict(scope='v110_same_cubin_two_passed_capacity_shells_not_real_trace_GEMM',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        compile_receipt_sha256=sha(directory/'analysis.json'),cubin_sha256=sha(directory/(STEM+'.cubin')),
        selected_modes=[0,1],rejected_scaled_shell_not_launched=True,static_gate_not_relaxed=True,
        sources={path:sha(ROOT/path) for path in ('scripts/run_o78_shell_capacity.py',
            'csrc/sm80/roof_o78_shell_capacity_driver.cpp')},commands=[],
        extension_sha256_before=sha(extensions[0]),production_default_changed=False)
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    executable=out/'capacity_driver'
    run([str(cuda/'nvcc'),'-std=c++17','-O2',str(ROOT/'csrc/sm80/roof_o78_shell_capacity_driver.cpp'),
        '-lcuda','-o',str(executable)],'driver_build.log')
    run(['nvidia-smi','--query-gpu=name,clocks.sm,temperature.gpu,power.draw','--format=csv'],'gpu_before.txt')
    run([str(executable),str(directory/(STEM+'.cubin')),'3'],'results.jsonl')
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    r['runtime']=summarize(rows)
    run(['nvidia-smi','--query-gpu=name,clocks.sm,temperature.gpu,power.draw','--format=csv'],'gpu_after.txt')
    r['extension_sha256_after']=sha(extensions[0])
    if r['extension_sha256_after']!=r['extension_sha256_before']:raise ValueError('production extension changed')
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='analysis.json'}
    (out/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['runtime'],indent=2),flush=True)


if __name__=='__main__':main()

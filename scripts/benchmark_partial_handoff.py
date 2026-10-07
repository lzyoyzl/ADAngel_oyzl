#!/usr/bin/env python3
"""v122 guarded warp handoff: numerical/sanitizer gate then full24 paired GEMM.

CPU classification is cached outside Events for this independent compute-only
experiment. It MUST NOT be reported as conversion or end-to-end performance.
No formal/default extension, source quantizer, trace or 5090 changes.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from inspect_partial_handoff import observe, ROOT, PROVENANCE
from probe_partial_handoff_codegen import SYMBOL, CONTROL, producer_header
from validate_unmeasured_eight_warp import Driver as OriginalDriver, SOURCE


def classify(case):
    """Separate candidate status; never overwrite the control's fullK guard."""
    import torch
    a=case.state['asq'].cpu().numpy().astype(np.int64)
    w=case.state['wsq'].cpu().numpy().astype(np.int64)
    old=case.state['status'].cpu().numpy().reshape(case.m//64,case.n//128)
    proof=observe(a,w,old)
    am=a.reshape(-1,64,32).max(1);wm=w.reshape(-1,128,32).max(1)
    narrow=(am[:,None,:]*wm[None,:,:]<=32767**2).all(-1)
    new=np.where((old==0)&~narrow,3,old).astype(np.uint32)
    if int((new==0).sum())!=proof['narrow_integer_ctas'] or int((new==3).sum())!=proof['wide_integer_ctas']:
        raise ValueError('classification and exact bound disagree')
    return torch.from_numpy(new.reshape(-1)).to(case.state['status'].device),proof


class Driver(OriginalDriver):
    def __init__(self,output,codegen):
        output.mkdir(parents=True)
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        receipt=json.loads((codegen/'codegen.json').read_text())
        if not receipt['compile_gate']['passed']:raise ValueError('compile investment gate failed')
        for path,digest in receipt['sources'].items():
            if sha(ROOT/path)!=digest:raise ValueError('source drift: '+path)
        for path,digest in {**receipt['artifact_sha256'],**receipt['headers']}.items():
            if sha(codegen/path)!=digest:raise ValueError('codegen artifact drift: '+path)
        cubin=codegen/'o78_partial_handoff.cubin'
        if sha(cubin)!=receipt['cubin_sha256'] or (codegen/'o78_partial_handoff_producer_generated.cuh').read_text()!=producer_header():
            raise ValueError('kernel or generated producer drift')
        library=output/'liblatency_validation.so'
        cmd=['g++','-std=c++17','-O2','-shared','-fPIC',str(SOURCE),
            '-I/usr/local/cuda-12.8/include','-L/usr/local/cuda-12.8/lib64/stubs','-lcuda','-o',str(library)]
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            env={**os.environ,'TMPDIR':str(ROOT/'tmp')})
        (output/'host_compile.log').write_text(result.stdout)
        if result.returncode:raise RuntimeError('driver compile failed')
        self.lib=ct.CDLL(str(library.resolve()));self.lib.roof_latency_error.restype=ct.c_char_p
        self.lib.roof_latency_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_int,ct.c_int,ct.c_uint,
            ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_int)]
        self.lib.roof_latency_close.argtypes=[ct.c_void_p]
        self.lib.roof_latency_benchmark.argtypes=[ct.c_void_p,ct.POINTER(ct.c_uint64)]+[ct.c_int]*6+[
            ct.c_void_p,ct.POINTER(ct.c_float)]
        self.handles={};self.resources={}
        try:
            for policy,symbol in enumerate((CONTROL,SYMBOL)):
                h=ct.c_void_p();values=(ct.c_int*4)();shared=(34304,51712)[policy]
                self.check(self.lib.roof_latency_open(str(cubin.resolve()).encode(),symbol.encode(),78,
                    (128,256)[policy],shared,ct.byref(h),values))
                self.handles['o78',policy]=h
                self.resources['o78',policy]=dict(registers=values[0],local_bytes=values[1],
                    threads=values[2],active_blocks=values[3],active_warps=values[2]*values[3]//32,
                    shared_bytes=shared,kernel_symbol=symbol)
            r=self.resources['o78',1]
            if r['active_blocks']<2 or r['threads']!=256 or r['local_bytes'] or r['registers']>128:
                raise ValueError('real resources fail the predeclared gate')
            self.receipt=dict(command=cmd,library_sha256=sha(library),codegen=receipt,
                resources={str(p):r for (_,p),r in self.resources.items()})
            (output/'resources.json').write_text(json.dumps(self.receipt,indent=2)+'\n')
        except Exception:self.close();raise

    def run_case(self,case,status,policy,warmup,repeats):
        tensors=tuple(case.state[k] for k in ('a','w','as','ws','af','wf','ab','wb'))+(
            case.state['status'] if policy==0 else status,case.state['y'])
        return self.run('o78',policy,tensors,case.m,case.n,warmup,repeats)


def numerical_pair(driver,case,status):
    import torch
    old,_=driver.run_case(case,status,0,0,1);old=old.clone()
    new,_=driver.run_case(case,status,1,0,1)
    if new.dtype!=torch.float32 or not bool(torch.isfinite(new).all()) or not torch.equal(
            old.view(torch.int32),new.view(torch.int32)):
        raise ValueError('candidate must equal guarded current best bitwise')
    return old


def validate(driver,prep,native):
    import torch
    from adangel.quantization import mixed_formats as mf
    from benchmark_o78_fused_prepare import Case
    checks=[];seen=set()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for variant in ('o7','o8'):
            wf,af=mf.VARIANTS[variant]
            for pattern in ('random','zero','saturated','fallback'):
                m,n=(128,256) if pattern in ('random','fallback') else (64,128)
                torch.manual_seed(20261007)
                a=(torch.randn(m,4096,device='cuda')*.4).half()
                w=(torch.randn(n,4096,device='cuda')*.1).half()
                if pattern=='zero':a.zero_();w.zero_()
                if pattern=='saturated':
                    a[:,::2]=-8;a[:,1::2]=7;w[:,::2]=-7;w[:,1::2]=6
                ws,acs=mf.quantize_source(w,wf),mf.quantize_source(a,af)
                if pattern=='fallback':
                    if variant=='o7':acs['scale'].fill_(127);acs['scale'][:,-1]=159
                    else:ws['scale'].fill_(1);ws['scale'][:,-1]=192
                base=native._benchmark_mixed(variant,'compute_only',ws,acs,0,1,2,
                    '64x128x256','group_major',59,5)
                case=Case(variant,ws,acs,base);guard=prep.prepare(case)
                status,proof=classify(case)
                old=numerical_pair(driver,case,status)
                torch.testing.assert_close(old,base['output'],rtol=1e-3,atol=1e-3)
                seen.update(int(x) for x in np.unique(status.cpu().numpy()))
                checks.append(dict(variant=variant,pattern=pattern,shape=[m,n,4096],guard=guard,
                    handoff_guard=proof,bitwise_best=True,production_reference_close=True,
                    finite_fp32=True,nondefault_stream=True))
    stream.synchronize()
    if not {0,1,3}.issubset(seen):raise ValueError('must execute narrow, FP32 and wide INT32 paths')
    return dict(passed=True,checks=checks,statuses_seen=sorted(seen),
        scope='small_MN_fullK_numerical_not_performance_screen')


def summarize(rows):
    from adangel.benchmark.metrics import bootstrap_median_ci
    ids=sorted({r['sample_id'] for r in rows});index={(r['sample_id'],r['variant'],r['round'],r['policy']):r for r in rows}
    if len(ids)!=24 or len(rows)!=288 or len(index)!=288 or set(index)!={
            (s,v,r,p) for s in ids for v in ('o7','o8') for r in range(3) for p in (0,1)}:
        raise ValueError('full24 three-round complete paired records required')
    summaries=[]
    for v in ('o7','o8'):
        ratios=[statistics.median(index[s,v,r,0]['summary']['median_ms']/index[s,v,r,1]['summary']['median_ms']
            for r in range(3)) for s in ids]
        for p in (0,1):
            selected=[index[s,v,r,p] for s in ids for r in range(3)]
            errors=[index[s,v,0,p]['mse_vs_reference'] for s in ids]
            summaries.append(dict(variant=v,policy=p,records=len(selected),samples=24,
                median_ms=statistics.median(statistics.median(index[s,v,r,p]['summary']['median_ms']
                    for r in range(3)) for s in ids),
                paired_speedup=statistics.median(ratios) if p else 1.,
                paired_ci95=list(bootstrap_median_ci(ratios,10000,.95,20261007)) if p else [1.,1.],
                cv_failed=sum(r['summary']['cv_percent']>=3 for r in selected),
                median_MSE=statistics.median(errors),mean_MSE=statistics.mean(errors)))
    return dict(scope='guarded_handoff_full24_cached_GEMM_only',summary=summaries,
        records=len(rows),bitwise_best=all(r['bitwise_best'] for r in rows),
        conversion_or_E2E_measured=False,no_filtering=True,production_default_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--codegen',type=Path,default=Path('reports/o378_roof_v122_handoff_codegen'))
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    import torch
    from adangel import _sm80 as native
    from adangel.trace.storage import sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,source_identity,mse
    from benchmark_a100_roof_trace import measurement_order
    from benchmark_a100_o1 import stats,command
    from benchmark_o78_row_fused import Driver as PrepDriver
    from benchmark_o78_fullk_gpu_prepare import checked_gpu_build
    from benchmark_o78_fused_prepare import Case
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    context_anchor=torch.empty(1,device='cuda')
    if torch.cuda.get_device_capability()!=(8,0):raise ValueError('A100 required')
    extension_hash=sha256_file(Path(native.__file__))
    driver=Driver(out/'build',args.codegen)
    lib,prep_receipt=checked_gpu_build(ROOT/'reports/o378_roof_v73_codegen')
    prep=PrepDriver(lib,ROOT/'reports/o378_roof_v67_codegen')
    def save(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    def append(n,x):
        with (out/n).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
    try:
        save('validation.json',validate(driver,prep,native))
        if args.validate_only:
            print('HANDOFF NUMERICAL VALIDATION PASSED',flush=True);return
        data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';rawdir=ROOT/'data/raw/llama2_7b_prefill'
        manifest,ph=inspect_inputs(data)
        raw,rh=inspect_raw_inputs(rawdir,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
        raw_index={r['sample_id']:r for r in raw['samples']}
        reference={(r['sample_id'],r['variant']):r for r in map(json.loads,PROVENANCE.read_text().splitlines())}
        if len(reference)!=48:raise ValueError('missing source authority')
        save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),torch=torch.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),extension_sha256=extension_hash,
            raw_manifest_sha256=rh,prepared_manifest_sha256=ph,resources=driver.receipt['resources'],
            source_hashes={str(s.relative_to(ROOT)):sha256_file(s) for s in (Path(__file__),SOURCE,
                ROOT/'scripts/inspect_partial_handoff.py')},preparation_receipt=prep_receipt,
            warmup=1000,repeats=200,rounds=3,conversion_inner_repeats=100,
            input_and_timing_contract='same_original_sources_single_stream_preallocated_interleaved',
            CPU_narrow_guard_cached_outside_Events=True,conversion_or_E2E_claim=False,
            no_filtering=True,no_clock_lock=True,production_default_changed=False))
        rows=[]
        for si,entry in enumerate(manifest['samples']):
            sid=entry['sample_id'];re=raw_index[sid]
            record=_load_and_validate_raw(rawdir/re['file'],re['layer'],re['projection'])
            for vi,v in enumerate(('o7','o8')):
                wf,af=mf.VARIANTS[v]
                ws=mf.quantize_source(record['weight_fp16'].cuda(),wf)
                acs=mf.quantize_source(record['activation_fp16'].cuda(),af)
                identity=dict(sample_id=sid,variant=v,raw_sha256=re['sha256'],
                    weight=source_identity(ws),activation=source_identity(acs))
                if identity!=reference[sid,v]:raise ValueError('source differs from measured v99')
                append('source_provenance.jsonl',identity)
                base=native._benchmark_mixed(v,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)
                case=Case(v,ws,acs,base);guard=prep.prepare(case);status,proof=classify(case)
                expected=numerical_pair(driver,case,status)
                fp=native._benchmark_mixed(mf.PAIRED_BASELINE[v],'compute_only',ws,acs,0,1,2,
                    '64x128x256','row_major')['output']
                error=mse(expected,fp)
                for ri in range(3):
                    append('gpu_snapshots.jsonl',dict(sample_id=sid,variant=v,round=ri,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order((0,1),si,vi,ri)
                    for policy in order:
                        y,times=driver.run_case(case,status,policy,1000,200)
                        if not bool(torch.isfinite(y).all()) or not torch.equal(y.view(torch.int32),expected.view(torch.int32)):
                            raise ValueError('full24 bitwise/finite regression')
                        current=mse(y,fp)
                        if current!=error:raise ValueError('MSE changed')
                        row=dict(sample_id=sid,variant=v,round=ri,policy=policy,mode='compute_only',order=list(order),
                            raw_ms=times,summary=stats(times),mse_vs_reference=current,reference=mf.PAIRED_BASELINE[v],
                            bitwise_best=True,finite_fp32=True,guard=guard,handoff_guard=proof,
                            conversion_included=False,CPU_narrow_guard_cached=True,
                            resource=driver.resources['o78',policy],total_timing='single_execution_cuda_event')
                        rows.append(row);append('results.jsonl',row)
                print(sid,v,'full24 paired complete',flush=True)
                del case,status,ws,acs,base,expected,fp,y
            del record
        if sha256_file(Path(native.__file__))!=extension_hash or sha256_file(data/'manifest.json')!=ph or sha256_file(rawdir/'trace_manifest.json')!=rh:
            raise ValueError('production extension or trace changed')
        result=summarize(rows);save('summary.json',result)
        print(json.dumps(result,indent=2),flush=True)
    finally:prep.close();driver.close()


if __name__=='__main__':main()

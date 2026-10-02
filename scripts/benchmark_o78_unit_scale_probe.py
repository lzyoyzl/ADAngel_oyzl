#!/usr/bin/env python3
"""v77 UNIT-scale cost isolation. Never report these times/MSE as original O7/O8.

Integer payloads are reused, but ALL real scale information is replaced by one.
The specialized diagnostic is correct only for this altered input problem.
No preprocessing/Cold/steady timings, acceptance or default promotion are made.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import benchmark_o78_fullk_integer_probe as original
from probe_o78_unit_scale_codegen import ROOT, SYMBOL, CONTROL, unit_header


def require_unit_values(arrays, status):
    if not arrays or any(not np.all(np.asarray(x)==1) for x in arrays) or not np.all(np.asarray(status)==0):
        raise ValueError('unit-scale diagnostic rejects non-unit scale/factor/base or nonzero status')


def checked(directory):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    if not r['diagnostic_only'] or r['performance_candidate'] or r['default_changed']:
        raise ValueError('diagnostic receipt required')
    for name, sha in r['sources'].items():
        if digest(ROOT/name)!=sha: raise ValueError('source drift: '+name)
    if digest(directory/'o78_unit_scale.cubin')!=r['cubin_sha256']:
        raise ValueError('binary drift')
    if digest(directory/'o78_unit_scale_generated.cuh')!=r['generated_header_sha256']:
        raise ValueError('generated header drift')
    if (directory/'o78_unit_scale_generated.cuh').read_text()!=unit_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()):
        raise ValueError('generator drift')
    if not r['control_opcode_counts_match'] or not r['control_instructions_match']:
        raise ValueError('recompiled control drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry ISA audit failed')
    return r


class Driver(original.Driver):
    def __init__(self, baseline, candidate):
        self.receipts=dict(baseline=original.checked(baseline), diagnostic=checked(candidate))
        super().__init__(baseline)
        self.check(self.lib.roof_probe_close(self.handles.pop(0)))
        self.handles[0]=self.handles.pop(1)
        self.resources={0:self.resources[1]}
        try:
            handle=ct.c_void_p()
            self.check(self.lib.roof_probe_open(str((candidate/'o78_unit_scale.cubin').resolve()).encode(),
                                               SYMBOL.encode(),34304,ct.byref(handle)))
            self.handles[1]=handle
            values=(ct.c_int*4)(); self.check(self.lib.roof_probe_resources(handle,values))
            self.resources[1]=dict(registers_per_thread=values[0],local_size_bytes=values[1],threads=values[2],
                active_blocks_per_sm=values[3],shared_memory_bytes=34304,cta_tile=[64,128,128],pipeline_stages=2)
            self.resources[0]['kernel_symbol']=CONTROL; self.resources[1]['kernel_symbol']=SYMBOL
        except Exception:
            self.close(); raise

    def run(self, policy, payload, scales, meta, warmup, repeats):
        # This verification is intentionally outside native CUDA Event intervals.
        # The isolated diagnostic is NEVER admitted for general experiment data.
        require_unit_values([x.cpu().numpy() for x in (*scales,*meta['cuda'][:-1])],meta['cuda'][-1].cpu().numpy())
        return super().run(policy,payload,scales,meta,warmup,repeats)


def make_case(aq,wq):
    import torch
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    if aq.ndim!=2 or wq.ndim!=2 or aq.dtype!=torch.int8 or wq.dtype!=torch.int8 or not aq.is_cuda or aq.device!=wq.device:
        raise ValueError('CUDA INT8 matrices required')
    m,k=aq.shape; n,wk=wq.shape
    if k!=4096 or wk!=k or m%64 or n%128 or int(wq.min()) < -8 or int(wq.max()) > 7:
        raise ValueError('aligned full-K INT8/INT4 data required')
    # At unit scales every product/prefix is bounded by 4096*128*8=2**22.
    # Both INT32 arithmetic and FP32 final output therefore represent integers exactly.
    a=split_int8_to_packed_int4(aq).reshape(2,m,32,64).permute(0,2,1,3).contiguous()
    w=pack_q4(wq).reshape(n,32,64).permute(1,0,2).contiguous()
    scales=tuple(torch.ones((32,rows),device=aq.device,dtype=torch.float32).T for rows in (m,n))
    factors=tuple(torch.ones((32,rows),device=aq.device,dtype=torch.int32) for rows in (m,n))
    bases=tuple(torch.ones(rows,device=aq.device,dtype=torch.float32) for rows in (m,n))
    flags=np.zeros((m//64)*(n//128),dtype=np.uint32)
    meta=dict(status_flat=flags,cuda=(*factors,*bases,torch.zeros(len(flags),device=aq.device,dtype=torch.int32)))
    return dict(packed_activation_g128_major=a,packed_weight_g128_major=w),scales,meta


def validate(driver):
    import torch
    checks=[]; stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ('random','zero','extrema','alternating'):
            torch.manual_seed(20261003)
            m,n=(128,256) if pattern=='extrema' else (64,128)
            a=torch.randint(-128,128,(m,4096),device='cuda',dtype=torch.int8)
            w=torch.randint(-8,8,(n,4096),device='cuda',dtype=torch.int8)
            if pattern=='zero': a.zero_(); w.zero_()
            if pattern=='extrema': a.fill_(-128); w.fill_(-8)
            if pattern=='alternating':
                a[:,::2]=-128; a[:,1::2]=127; w[:,::2]=7; w[:,1::2]=-8
            case=make_case(a,w); reference=(a.double()@w.double().T).float()
            for policy in (0,1):
                y,_=driver.run(policy,*case,0,1)
                assert torch.equal(y.view(torch.int32),reference.view(torch.int32))
                checks.append(dict(pattern=pattern,policy=policy,shape=[m,n,4096],exact_fp64_reference=True,
                                   finite_fp32=bool(torch.isfinite(y).all()),nondefault_stream=True))
        # Invalid input must be rejected before calling even the original native driver.
        case[2]['cuda'][0][0,0]=2
        try: driver.run(1,*case,0,1)
        except ValueError as exc: assert 'rejects non-unit' in str(exc)
        else: raise AssertionError('non-unit source admitted to diagnostic')
        stream.synchronize()
    return dict(passed=True,checks=checks,count=len(checks),nonunit_rejected=True,
                scope='unit_scales_only_small_MN_K4096_not_original_experiment_validation')


def summarize(rows):
    result=[]
    for variant in ('o7','o8'):
        selected=[r for r in rows if r['payload_origin']==variant]
        index={(r['sample_id'],r['round'],r['policy']):r for r in selected}
        ids=sorted({r['sample_id'] for r in selected}); rounds=sorted({r['round'] for r in selected})
        if len(index)!=len(selected) or set(index)!={(s,r,p) for s in ids for r in rounds for p in (0,1)}:
            raise ValueError('incomplete diagnostic pairing')
        for policy in (0,1):
            values=[index[s,r,policy] for s in ids for r in rounds]
            latency=[statistics.median(index[s,r,policy]['summary']['median_ms'] for r in rounds) for s in ids]
            ratios=[statistics.median(index[s,r,0]['summary']['median_ms']/index[s,r,policy]['summary']['median_ms'] for r in rounds) for s in ids]
            result.append(dict(payload_origin=variant,policy=policy,samples=len(ids),records=len(values),
                median_ms=statistics.median(latency),paired_ratio=statistics.median(ratios),
                cv_failed=sum(r['summary']['cv_percent']>=3 for r in values),
                exact_unit_reference=all(r['unit_reference_mse']==0 for r in values),
                interpretation='unit-scale_diagnostic_NOT_original_O7_O8_speedup_or_MSE'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--cubins',type=Path,default=Path('reports/o378_roof_v77_codegen'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,choices=(4,),default=4)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT) or min(a.rounds,a.repeats)<1 or a.warmup<0:
        p.error('fresh repository output and valid measurement parameters required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command,stats
    from benchmark_a100_roof_trace import measurement_order
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    anchor=torch.empty(1,device='cuda'); driver=Driver(a.baseline,a.cubins)
    a.output.mkdir(parents=True)
    def save(name,x): (a.output/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    def append(name,x):
        with (a.output/name).open('a') as f: f.write(json.dumps(x,allow_nan=False)+'\n')
    try:
        save('validation.json',validate(driver))
        if a.validate_only:
            print('UNIT-SCALE DIAGNOSTIC VALIDATION PASSED',flush=True);return
        data=ROOT/'data/prepared/llama2_7b_prefill_o0_o4';raw=ROOT/'data/raw/llama2_7b_prefill'
        manifest,mh=original.inspect_inputs(data)
        raw_manifest,rh=original.inspect_raw_inputs(raw,manifest,ROOT/'configs/trace/llama2_7b_prefill.yaml')
        raw_index={x['sample_id']:x for x in raw_manifest['samples']}
        save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),codegen=driver.receipts,
            resources=driver.resources,extension_sha256=sha256_file(Path(native.__file__)),
            raw_manifest_sha256=rh,prepared_manifest_sha256=mh,torch=torch.__version__,cuda=torch.version.cuda,
            diagnostic_only=True,all_scales_replaced_by_one=True,production_default_changed=False,no_filtering=True,
            not_an_original_experiment_speedup=True,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
        rows=[]
        for si,entry in enumerate(manifest['samples'][:a.samples]):
            re=raw_index[entry['sample_id']];rp=raw/re['file'];pp=data/entry['file']
            assert sha256_file(rp)==re['sha256'] and sha256_file(pp)==entry['sha256']
            record=_load_and_validate_raw(rp,re['layer'],re['projection'])
            prepared=load_prepared(pp,device='cpu')
            original.verify_raw_prepared(prepared,(record['activation_fp16'],record['weight_fp16']));del prepared
            for vi,(variant,(wf,af)) in enumerate(mf.VARIANTS.items()):
                ws=mf.quantize_source(record['weight_fp16'].cuda(),wf)
                acs=mf.quantize_source(record['activation_fp16'].cuda(),af)
                aq,_=mf.to_fixed_reference(acs);wq,_=mf.to_fixed_reference(ws)
                append('source_provenance.jsonl',dict(sample_id=entry['sample_id'],payload_origin=variant,
                    original_weight=original.source_identity(ws),original_activation=original.source_identity(acs),
                    raw_sha256=re['sha256'],scale_change='ALL replaced by one; diagnostic only'))
                case=make_case(aq,wq);reference=(aq.double()@wq.double().T).float()
                for ri in range(a.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=entry['sample_id'],payload_origin=variant,round=ri,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order((0,1),si,vi,ri)
                    for policy in order:
                        y,times=driver.run(policy,*case,a.warmup,a.repeats)
                        assert torch.equal(y.view(torch.int32),reference.view(torch.int32))
                        item=dict(sample_id=entry['sample_id'],payload_origin=variant,policy=policy,round=ri,
                            execution_order=list(order),raw_ms=times,summary=stats(times),unit_reference_mse=original.mse(y,reference),
                            finite_fp32=True,resources=driver.resources[policy],diagnostic_only=True,
                            mode='cached_UNIT_scale_core_NOT_original_compute_only')
                        rows.append(item);append('results.jsonl',item)
                print(entry['sample_id'],variant,'UNIT-scale diagnostic complete',flush=True)
            save('summary.json',dict(diagnostic_only=True,original_experiment_result=False,records=summarize(rows)))
        assert sha256_file(data/'manifest.json')==mh and sha256_file(raw/'trace_manifest.json')==rh
        print('UNIT-SCALE DIAGNOSTIC PAIRED TEST PASSED',flush=True)
    finally: driver.close()


if __name__=='__main__':main()

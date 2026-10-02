#!/usr/bin/env python3
"""v67 cached-factor independent A/B screen, NOT conversion or E2E evidence."""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from benchmark_a100_o1 import command, stats
from benchmark_a100_roof_trace import measurement_order
from benchmark_a100_roof_candidates import group_major_scales
from benchmark_a100_mixed import validate_fp16_result
from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, mse, source_identity, verify_raw_prepared
from o78_fullk_integer_metadata import prepare_fullk_metadata, dyadic_scale
from roof_payload_validation import verify_grouped_payload
from roof_reduction_validation import reference_fp64, mse_regression_ok

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ('adangel_roof_o78_fullk_control', 'adangel_roof_o78_fullk_candidate')


def checked(directory):
    from adangel.trace.storage import sha256_file
    result = json.loads((directory/'codegen.json').read_text())
    for name, digest in result['sources'].items():
        if sha256_file(ROOT/name) != digest: raise ValueError('source drift')
    for filename, field in (('o78_fullk.cubin','cubin_sha256'),('libo78_fullk_driver.so','driver_sha256')):
        if sha256_file(directory/filename) != result[field]: raise ValueError('artifact drift')
    if set(result['entries']) != set(SYMBOLS) or not all(
        x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma'] and x['all_copies_bypass_l1']
        for x in result['entries'].values()): raise ValueError('ISA audit failed')
    return result


class Driver:
    def __init__(self, directory):
        self.lib = ct.CDLL(str((directory/'libo78_fullk_driver.so').resolve()))
        self.lib.roof_probe_error.restype = ct.c_char_p
        self.lib.roof_probe_open.argtypes = [ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
        self.lib.roof_probe_close.argtypes = [ct.c_void_p]
        self.lib.roof_probe_resources.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int)]
        self.lib.roof_o78_fullk_benchmark.argtypes = ([ct.c_void_p]+[ct.c_uint64]*10+
            [ct.c_int]*5+[ct.c_void_p,ct.POINTER(ct.c_float)])
        self.handles = {}; self.resources = {}
        try:
            for policy, symbol in enumerate(SYMBOLS):
                handle = ct.c_void_p()
                self.check(self.lib.roof_probe_open(str((directory/'o78_fullk.cubin').resolve()).encode(),
                    symbol.encode(),34304,ct.byref(handle)))
                self.handles[policy] = handle
                values = (ct.c_int*4)(); self.check(self.lib.roof_probe_resources(handle,values))
                assert values[2] == 128
                self.resources[policy] = dict(registers_per_thread=values[0],local_size_bytes=values[1],
                    threads=values[2],active_blocks_per_sm=values[3],shared_memory_bytes=34304,
                    cta_tile=[64,128,128],pipeline_stages=2)
        except Exception: self.close(); raise

    def check(self, status):
        if status: raise RuntimeError(self.lib.roof_probe_error().decode())

    def close(self):
        for handle in self.handles.values(): self.check(self.lib.roof_probe_close(handle))
        self.handles.clear()

    def run(self, policy, best, scales, metadata, warmup, repeats):
        import torch
        a, w = best['packed_activation_g128_major'], best['packed_weight_g128_major']
        m, n = a.shape[2], w.shape[1]
        assert a.shape == (2,32,m,64) and w.shape == (32,n,64)
        assert m%64 == n%128 == 0 and all(t.is_cuda and t.is_contiguous() for t in (a,w))
        asc, wsc = scales
        assert asc.shape == (m,32) and wsc.shape == (n,32)
        assert asc.stride() == (1,m) and wsc.stride() == (1,n)
        assert all(t.is_cuda and t.dtype == torch.float32 for t in scales)
        assert not np.any(metadata['status_flat'] > 1), 'invalid inputs never expose unwritten output'
        tensors = metadata['cuda']
        assert all(t.is_cuda and t.is_contiguous() and t.device==a.device for t in tensors)
        y = torch.empty((m,n),dtype=torch.float32,device=a.device)
        times = (ct.c_float*repeats)()
        self.check(self.lib.roof_o78_fullk_benchmark(self.handles[policy],
            a.data_ptr(),w.data_ptr(),asc.data_ptr(),wsc.data_ptr(),
            *(t.data_ptr() for t in tensors),y.data_ptr(),m,n,4096,warmup,repeats,
            torch.cuda.current_stream().cuda_stream,times))
        if not torch.isfinite(y).all(): raise AssertionError('nonfinite output')
        return y,list(times)


def prepare(a_codes,w_codes,aq,wq,variant,tensor_scale):
    import torch
    start = time.perf_counter()
    ak,wk = ('ue8m0','e4m3') if variant=='o7' else ('e4m3','e6m2')
    squares = lambda q: q.reshape(q.shape[0],32,128).long().square().sum(-1).cpu().numpy()
    metadata = prepare_fullk_metadata(a_codes,w_codes,squares(aq),squares(wq),
        activation_kind=ak,weight_kind=wk,
        activation_base_multiplier=4 if variant=='o7' else np.float32(tensor_scale)/4,
        weight_base_multiplier=np.float32(tensor_scale) if variant=='o7' else 1)
    metadata['cuda'] = tuple(torch.from_numpy(metadata[name]).to(aq.device,
        dtype=torch.int32 if name=='status_flat' else None) for name in (
        'activation_factors','weight_factors','activation_bases','weight_bases','status_flat'))
    torch.cuda.current_stream().synchronize()
    metadata['preparation_wall_ms'] = (time.perf_counter()-start)*1000
    return metadata


def guard_record(meta):
    return dict(integer_ctas=int(np.sum(meta['status_flat']==0)),
        fallback_ctas=int(np.sum(meta['status_flat']==1)),invalid_ctas=int(np.sum(meta['status_flat']==2)),
        ctas=int(meta['status_flat'].size),preparation_wall_ms=meta['preparation_wall_ms'],
        metadata_bytes=sum(t.numel()*t.element_size() for t in meta['cuda']),
        scope='CPU_oracle_and_H2D_preparation_excluded_cached_GEMM_only')


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from validate_a100_split_grouped import pack_q4
    results = []
    for variant in ('o7','o8'):
        ak,wk = ('ue8m0','e4m3') if variant=='o7' else ('e4m3','e6m2')
        for pattern in ('random','zero','extrema','zero_scale'):
            m,n,k = 64,128,4096; torch.manual_seed(20261002)
            al,wl = (112,6) if variant=='o7' else (30,7)
            aq = torch.randint(-al,al+1,(m,k),device='cuda',dtype=torch.int8)
            wq = torch.randint(-wl,wl+1,(n,k),device='cuda',dtype=torch.int8)
            if pattern=='zero': aq.zero_();wq.zero_()
            if pattern=='extrema': aq[:,::2]=-al;aq[:,1::2]=al;wq[:,::2]=-wl;wq[:,1::2]=wl
            ac = (np.arange(m*32).reshape(m,32)%4+(121 if variant=='o7' else 48)).astype(np.uint8)
            wc = (np.arange(n*32).reshape(n,32)%4+(48 if variant=='o7' else 192)).astype(np.uint8)
            if pattern=='zero_scale':
                (wc if variant=='o7' else ac)[:,::2]=0
            am,wm = (4,.75) if variant=='o7' else (.75/4,1)
            def scales(codes,kind,mult):
                natural = np.array([[math.ldexp(mant,exp)*mult for mant,exp in
                    (dyadic_scale(int(c),kind) for c in row)] for row in codes],dtype=np.float32)
                return group_major_scales(torch.from_numpy(natural).cuda())
            asc,wsc = scales(ac,ak,am),scales(wc,wk,wm)
            values = (split_int8_to_packed_int4(aq),asc,pack_q4(wq),wsc)
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                best = native._benchmark_roof_candidate(variant,59,*values,0,1)
                verify_grouped_payload(best,59,values[0],values[2],values[3])
                meta = prepare(ac,wc,aq,wq,variant,.75)
                semantic = reference_fp64(variant,values)
                for fallback in (False,True):
                    if fallback:
                        meta['status_flat'].fill(1);meta['cuda'][-1].fill_(1)
                    for policy in (0,1):
                        y,_ = driver.run(policy,best,(asc,wsc),meta,0,1)
                        torch.testing.assert_close(y.double(),semantic,rtol=1e-3,atol=1e-3)
                        equal = torch.equal(y.view(torch.int32),best['output'].view(torch.int32))
                        if policy==0 or fallback: assert equal
                        results.append(dict(variant=variant,pattern=pattern,force_fallback=fallback,policy=policy,
                            bitwise_equal_best=equal,max_abs_difference=(y-best['output']).abs().max().item(),
                            mse_vs_best=mse(y,best['output']),**guard_record(meta)))
            stream.synchronize()
        print(variant,'synthetic semantic and fallback checks passed',flush=True)
    return dict(passed=True,count=len(results),checks=results)


def summarize(rows):
    from compare_roof_trace_candidates import metrics
    output = []
    for variant in ('o7','o8'):
        selected = [r for r in rows if r['variant']==variant]
        index = {(r['sample_id'],r['round'],r['policy']):r for r in selected}
        ids = sorted({r['sample_id'] for r in selected}); rounds = sorted({r['round'] for r in selected})
        assert len(index)==len(selected) and set(index)=={(s,r,p) for s in ids for r in rounds for p in (0,1)}
        for policy in (0,1):
            chosen = [r for r in selected if r['policy']==policy]
            lat = [statistics.median(index[s,r,policy]['summary']['median_ms'] for r in rounds) for s in ids]
            speed = [statistics.median(index[s,r,0]['summary']['median_ms']/index[s,r,policy]['summary']['median_ms'] for r in rounds) for s in ids]
            errors = [index[s,0,policy]['mse_vs_paired_fp16'] for s in ids]
            output.append(dict(variant=variant,policy=policy,samples=len(ids),records=len(chosen),
                median_ms=statistics.median(lat),paired_speedup=statistics.median(speed),
                paired_speedup_ci95=list(metrics.bootstrap_median_ci(speed,10000,.95,20261002)),
                median_mse=statistics.median(errors),mean_mse=statistics.mean(errors),
                cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in chosen),
                max_abs_vs_best=max(r['max_abs_vs_best'] for r in chosen),
                max_mse_vs_best=max(r['mse_vs_best'] for r in chosen)))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cubins',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,choices=(4,24),default=4);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT) or min(a.rounds,a.repeats)<1 or a.warmup<0:
        p.error('fresh repository output and valid measurements required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    torch.cuda.init();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_capability()==(8,0)
    # Driver API needs a current context; lazy CUDA init alone is insufficient.
    torch.empty(1,device='cuda')
    codegen = checked(a.cubins); a.output.mkdir(parents=True)
    def save(name,value): (a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (a.output/name).open('a') as f: f.write(json.dumps(value,allow_nan=False)+'\n')
    driver = Driver(a.cubins)
    try:
        save('validation.json',validate(driver))
        if a.validate_only: return
        manifest,mh = inspect_inputs(a.data); raw,rh = inspect_raw_inputs(a.raw_data,manifest,a.trace_config)
        raw_index = {r['sample_id']:r for r in raw['samples']}
        save('environment.json',dict(source_commit=command('git','rev-parse','HEAD'),
            extension_sha256=sha256_file(Path(native.__file__)),codegen=codegen,
            resources=driver.resources,prepared_manifest_sha256=mh,raw_manifest_sha256=rh,
            scope='cached_compute_only_not_conversion_or_Cold_or_steady_state',
            policy='unlocked_shared_GPU_cyclic_AB_all_200_samples_preserved',
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
        rows=[]
        for si,entry in enumerate(manifest['samples'][:a.samples]):
            re=raw_index[entry['sample_id']]; path=a.data/entry['file'];rp=a.raw_data/re['file']
            assert sha256_file(path)==entry['sha256'] and sha256_file(rp)==re['sha256']
            x=load_prepared(path,device='cuda');record=_load_and_validate_raw(rp,re['layer'],re['projection'])
            operands=(record['activation_fp16'],record['weight_fp16']);verify_raw_prepared(x,operands)
            for vi,(variant,(wf,af)) in enumerate(mf.VARIANTS.items()):
                wsrc=mf.quantize_source(operands[1].cuda(),wf);asrc=mf.quantize_source(operands[0].cuda(),af)
                append('source_provenance.jsonl',dict(sample_id=x.sample_id,variant=variant,
                    weight=source_identity(wsrc),activation=source_identity(asrc)))
                paired_case=mf.PAIRED_BASELINE[variant]
                fp=native._benchmark_mixed(paired_case,'compute_only',wsrc,asrc,0,1,100,'64x128x256','row_major')
                validate_fp16_result(fp,wsrc,asrc);paired=fp['output']
                base=native._benchmark_mixed(variant,'compute_only',wsrc,asrc,0,1,100,'64x128x256','group_major')
                values=(*base['converted_activation'],*base['converted_weight'])
                best=native._benchmark_roof_candidate(variant,59,*values,0,1)
                payload=verify_grouped_payload(best,59,values[0],values[2],values[3])
                aq,_=mf.to_fixed_reference(asrc);wq,_=mf.to_fixed_reference(wsrc)
                tensor_scale=float((wsrc if variant=='o7' else asrc)['tensor_scale'].item())
                meta=prepare(asrc['scale'].cpu().numpy(),wsrc['scale'].cpu().numpy(),aq,wq,variant,tensor_scale)
                baseline_mse=mse(best['output'],paired)
                for r in range(a.rounds):
                    append('gpu_snapshots.jsonl',dict(sample_id=x.sample_id,variant=variant,round=r,time=time.time(),
                        gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                    order=measurement_order((0,1),si,vi,r)
                    for policy in order:
                        y,times=driver.run(policy,best,(values[1],values[3]),meta,a.warmup,a.repeats)
                        equal=bool(torch.equal(y.view(torch.int32),best['output'].view(torch.int32)))
                        if policy==0: assert equal
                        torch.testing.assert_close(y,best['output'],rtol=1e-3,atol=1e-3)
                        error=mse(y,paired)
                        assert mse_regression_ok(error,baseline_mse), 'MSE regression beyond allowed tolerance'
                        row=dict(sample_id=x.sample_id,variant=variant,round=r,policy=policy,execution_order=order,
                            raw_ms=times,summary=stats(times),bitwise_equal_best=equal,
                            max_abs_vs_best=(y-best['output']).abs().max().item(),mse_vs_best=mse(y,best['output']),
                            mse_vs_paired_fp16=error,current_best_mse=baseline_mse,paired_fp16=paired_case,
                            finite_fp32=True,output_close_best=True,MSE_regression_passed=True,
                            guard=guard_record(meta),resources=driver.resources[policy],**payload)
                        rows.append(row);append('results.jsonl',row)
                print(x.sample_id,variant,'complete',guard_record(meta),flush=True)
        assert len(rows)==a.samples*2*a.rounds*2
        assert sha256_file(a.data/'manifest.json')==mh and sha256_file(a.raw_data/'trace_manifest.json')==rh
        result=dict(scope='cached_compute_only_host_guard_excluded_not_E2E',production_default_changed=False,
            correctness_passed=True,no_filtering=True,records=summarize(rows))
        save('summary.json',result);print(json.dumps(result,indent=2),flush=True)
    finally: driver.close()


if __name__=='__main__':main()

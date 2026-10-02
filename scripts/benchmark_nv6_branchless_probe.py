#!/usr/bin/env python3
"""v66 vector16 old/new paired FP6 activation conversion; unchanged O8 GEMM.

Measures one conversion only, not conversion-total, GEMM or Cold/steady.
Preserves all raw timings/CV failures and checks packed values/scales/output.
"""
import argparse
import json
from pathlib import Path
import time
from benchmark_a100_o1 import command,stats
from benchmark_vector_conversion_probe import summarize
from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,source_identity,mse
from vector_conversion_probe import VectorConversionProbe

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,choices=(4,24),default=4);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT) or min(a.rounds,a.repeats,a.inner)<1 or a.warmup<0:
        p.error('fresh repository output and valid counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.trace.storage import sha256_file,load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    if torch.cuda.get_device_capability()!=(8,0):raise RuntimeError('A100 SM80 required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    audit=a.directory/('audit_rechecked.json' if (a.directory/'audit_rechecked.json').exists() else 'audit.json')
    receipt=json.loads(audit.read_text())
    if not receipt['passed']:raise RuntimeError('conversion audit must pass')
    libraries=[a.directory/f'policy_{i}/libconversion.so' for i in (0,1)]
    for i,lib in enumerate(libraries):
        if sha256_file(lib)!=receipt['builds'][i]['sha256']:raise ValueError('library identity changed')
    probes=[VectorConversionProbe(f) for f in libraries]
    manifest,mh=inspect_inputs(a.data);raw_manifest,rh=inspect_raw_inputs(a.raw_data,manifest,a.trace_config)
    raw_index={e['sample_id']:e for e in raw_manifest['samples']}
    a.output.mkdir(parents=True)
    def save(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    def append(name,obj):
        with (a.output/name).open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    save('environment.json',dict(source_commit=command('git','rev-parse','HEAD'),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,binary_sha256=sha256_file(Path(native.__file__)),
        libraries=[dict(file=str(f),sha256=sha256_file(f)) for f in libraries],audit_sha256=sha256_file(audit),
        raw_manifest_sha256=rh,prepared_manifest_sha256=mh,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        scope='only_FP6_activation_conversion_not_total_or_GEMM_or_E2E',
        timing='single_stream_native_cuda_events_inner_amortized; alternate sample/round; shared GPU; unlocked; no filtering',
        gemm_tune=59,conversion_vector_elements=16,production_changed=False))
    records=[];checks=[]
    wf,af=mf.VARIANTS['o8']
    for si,entry in enumerate(manifest['samples'][:a.samples]):
        raw_entry=raw_index[entry['sample_id']];sid=entry['sample_id']
        if sha256_file(a.data/entry['file'])!=entry['sha256'] or sha256_file(a.raw_data/raw_entry['file'])!=raw_entry['sha256']:
            raise ValueError('input file changed')
        raw=_load_and_validate_raw(a.raw_data/raw_entry['file'],raw_entry['layer'],raw_entry['projection'])
        prepared=load_prepared(a.data/entry['file'],device='cpu')
        verify_raw_prepared(prepared,(raw['activation_fp16'],raw['weight_fp16']));del prepared
        ws=mf.quantize_source(raw['weight_fp16'].cuda(),wf);acs=mf.quantize_source(raw['activation_fp16'].cuda(),af)
        append('source_provenance.jsonl',dict(sample_id=sid,weight=source_identity(ws),activation=source_identity(acs)))
        q,sc=mf.to_fixed_reference(acs)
        expected=split_int8_to_packed_int4(q).reshape(2,4096,32,64).permute(0,2,1,3).contiguous()
        converted={}
        for ri in range(a.rounds):
            append('gpu_snapshots.jsonl',dict(sample_id=sid,round=ri,time=time.time(),
                gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
            for impl in ((0,1) if (si+ri)%2==0 else (1,0)):
                r=probes[impl](acs,2,a.warmup,a.repeats,a.inner)
                assert torch.equal(r['packed'],expected),(sid,impl,'payload')
                assert torch.equal(r['scale'].contiguous().view(torch.int32),sc.view(torch.int32)),(sid,impl,'scale')
                converted[impl]=r
                record=dict(sample_id=sid,format=af,round=ri,implementation=impl,summary=stats(r['timings_ms']),
                    timings_ms=r['timings_ms'],inner=a.inner,bitwise_payload_and_scale=True,conversion_kernel_count=1)
                records.append(record);append('results.jsonl',record)
                print(sid,ri,impl,record['summary']['median_ms'],record['summary']['cv_percent'],flush=True)
        fp=native._benchmark_mixed('o6','compute_only',ws,acs,0,1,2,'64x128x256','row_major')['output']
        base=native._benchmark_mixed('o8','compute_only',ws,acs,0,1,2,'64x128x256','group_major',59,5)['output']
        wresult=probes[0](ws,2,0,1,2)
        wn=wresult['packed'].permute(1,0,2).contiguous().reshape(4096,2048)
        for impl in (0,1):
            r=converted[impl]
            an=r['packed'].permute(0,2,1,3).contiguous().reshape(8192,2048)
            y=native._benchmark_roof_candidate('o8',59,an,r['scale'],wn,wresult['scale'],0,1)['output']
            assert y.dtype==torch.float32 and torch.isfinite(y).all()
            assert torch.equal(y.view(torch.int32),base.view(torch.int32))
            check=dict(sample_id=sid,variant='o8',implementation=impl,output_bitwise_current_best=True,
                mse_vs_current_best=mse(y,base),mse_vs_paired_fp16=mse(y,fp),paired_reference='o6')
            checks.append(check);append('output_mse.jsonl',check)
        save('summary.json',summarize(records));save('validation.json',dict(passed=True,records=len(records),
            samples=si+1,output_checks=len(checks),gemm_performance_measured=False,conversion_only=True))
        del raw,ws,acs,q,sc,expected,converted,fp,base,wresult,wn,an,y,r
    print('NV6 BRANCHLESS SCREEN PASSED',flush=True)


if __name__=='__main__':main()

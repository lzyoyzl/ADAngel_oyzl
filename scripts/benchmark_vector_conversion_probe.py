#!/usr/bin/env python3
"""v53 paired vector conversion screening against the best scalar controls; never report these times as E2E."""
import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_a100_o1 import command, stats
from benchmark_a100_mixed_trace import (
    inspect_inputs, inspect_raw_inputs, verify_raw_prepared, source_identity, mse,
)


def order(implementations, sample, fmt, round_index):
    offset=(sample+fmt+round_index)%len(implementations)
    return implementations[offset:]+implementations[:offset]


def summarize(records):
    from adangel.benchmark.metrics import bootstrap_median_ci
    index={(r['sample_id'],r['format'],r['round'],r['implementation']):r for r in records}
    if len(index)!=len(records): raise ValueError('duplicate measurements')
    if not records: raise ValueError('empty measurements')
    policies={r['implementation'] for r in records}
    rounds={r['round'] for r in records}
    identities={(r['sample_id'],r['format']) for r in records}
    if 0 not in policies or rounds!=set(range(len(rounds))) or set(index)!={(sid,fmt,r,i) for sid,fmt in identities for r in rounds for i in policies}:
        raise ValueError('incomplete paired coverage')
    summary=[]
    for fmt,impl in sorted({(r['format'],r['implementation']) for r in records}):
        rows=[r for r in records if (r['format'],r['implementation'])==(fmt,impl)]
        samples=[]
        for sid in sorted({r['sample_id'] for r in rows}):
            selected=[r for r in rows if r['sample_id']==sid]
            ratios=[index[(sid,fmt,r['round'],0)]['summary']['median_ms']/r['summary']['median_ms'] for r in selected]
            samples.append(dict(sample_id=sid,
                median_ms=statistics.median(r['summary']['median_ms'] for r in selected),
                paired_speedup=statistics.median(ratios)))
        speeds=[r['paired_speedup'] for r in samples]
        summary.append(dict(format=fmt,implementation=impl,samples=len(samples),records=len(rows),
            median_ms=statistics.median(r['median_ms'] for r in samples),
            paired_speedup=statistics.median(speeds),
            paired_speedup_ci95=list(bootstrap_median_ci(speeds,10000,.95,20261001)) if len(samples)>1 else None,
            cv_failed_records=sum(r['summary']['cv_percent']>=3 for r in rows),per_sample=samples))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--samples',type=int,default=24)
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=200)
    p.add_argument('--inner',type=int,default=100)
    p.add_argument('--implementations',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--data',type=Path,default=Path('data/prepared/llama2_7b_prefill_o0_o4'))
    p.add_argument('--raw-data',type=Path,default=Path('data/raw/llama2_7b_prefill'))
    p.add_argument('--trace-config',type=Path,default=Path('configs/trace/llama2_7b_prefill.yaml'))
    args=p.parse_args()
    if (args.output.exists() or not 1<=args.samples<=24 or args.rounds<1 or args.warmup<0 or args.repeats<2
        or args.inner<2 or 0 not in args.implementations or len(set(args.implementations))!=len(args.implementations)
        or any(i not in (0,1,2) for i in args.implementations)):
        p.error('fresh output, baseline0, unique implementations0..2 and positive counts required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.trace.storage import load_prepared,sha256_file
    from adangel.trace.prepare import _load_and_validate_raw
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('A100 SM80 required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True)
    def save(name,value): (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def append(name,value):
        with (args.output/name).open('a') as out: out.write(json.dumps(value,allow_nan=False)+'\n')
    manifest_hash=raw_hash=None
    if args.synthetic:
        entries=[dict(sample_id='synthetic_seed_61001')]
    else:
        manifest,manifest_hash=inspect_inputs(args.data)
        raw_manifest,raw_hash=inspect_raw_inputs(args.raw_data,manifest,args.trace_config)
        raw_index={e['sample_id']:e for e in raw_manifest['samples']}
        entries=manifest['samples'][:args.samples]
    save('environment.json',dict(git_commit=command('git','rev-parse','HEAD'),
        binary_sha256=sha256_file(Path(native.__file__)),library_sha256=sha256_file(args.library),torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),prepared_manifest_sha256=manifest_hash,raw_manifest_sha256=raw_hash,
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scope='standalone_conversion_probe',timing='native_cuda_event_inner_amortized',
        gemm_performance_measured=False,formal_dispatch_changed=False,
        policy='cyclic sample/format/round; shared GPU; no clock lock; no filtering'))
    from vector_conversion_probe import VectorConversionProbe
    probe=VectorConversionProbe(args.library)
    records=[];correctness=[]
    for si,entry in enumerate(entries):
        sid=entry['sample_id']
        if args.synthetic:
            torch.manual_seed(61001)
            raw={'activation_fp16':torch.randn(4096,4096,dtype=torch.float16)*.3,
                 'weight_fp16':torch.randn(4096,4096,dtype=torch.float16)*.02}
        else:
            raw_entry=raw_index[sid]
            if sha256_file(args.data/entry['file'])!=entry['sha256'] or sha256_file(args.raw_data/raw_entry['file'])!=raw_entry['sha256']:
                raise ValueError('input changed after preflight')
            x=load_prepared(args.data/entry['file'],device='cpu')
            raw=_load_and_validate_raw(args.raw_data/raw_entry['file'],raw_entry['layer'],raw_entry['projection'])
            verify_raw_prepared(x,(raw['activation_fp16'],raw['weight_fp16']))
            del x
        sources={};converted={}
        for fi,fmt in enumerate(mf.FORMATS):
            weight=mf.FORMATS[fmt][1]==4
            src=mf.quantize_source(raw['weight_fp16' if weight else 'activation_fp16'].cuda(),fmt)
            sources[fmt]=src
            append('source_provenance.jsonl',dict(sample_id=sid,format=fmt,source=source_identity(src)))
            q,sc=mf.to_fixed_reference(src)
            rows,k=src['shape'];groups=k//128
            packed=mf._pack_nibbles(q.to(torch.uint8)&15) if weight else split_int8_to_packed_int4(q)
            expected=(packed.reshape(rows,groups,64).permute(1,0,2) if weight else
                      packed.reshape(2,rows,groups,64).permute(0,2,1,3)).contiguous()
            converted[fmt]={}
            for ri in range(args.rounds):
                append('gpu_snapshots.jsonl',dict(sample_id=sid,format=fmt,round=ri,time=time.time(),
                    gpu=command('nvidia-smi','--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu','--format=csv')))
                for impl in order(args.implementations,si,fi,ri):
                    result=probe(src,impl,args.warmup,args.repeats,args.inner)
                    assert torch.equal(result['packed'],expected),(sid,fmt,impl,'packed')
                    assert torch.equal(result['scale'].contiguous().view(torch.int32),sc.contiguous().view(torch.int32)),(sid,fmt,impl,'scale')
                    converted[fmt][impl]=(result['packed'],result['scale'])
                    record=dict(sample_id=sid,format=fmt,round=ri,implementation=impl,
                        summary=stats(result['timings_ms']),timings_ms=list(result['timings_ms']),
                        inner=args.inner,bitwise_payload_and_scale=True,conversion_kernel_count=result['conversion_kernel_count'])
                    records.append(record);append('results.jsonl',record)
                    print(sid,fmt,ri,impl,record['summary']['median_ms'],flush=True)
            del q,sc,packed,expected,result
        for variant,(wf,af) in mf.VARIANTS.items():
            if variant not in ('o7','o8'): continue
            ws,acs=sources[wf],sources[af]
            fp=native._benchmark_mixed(mf.PAIRED_BASELINE[variant],'compute_only',ws,acs,0,1,2,'64x128x256','row_major')['output']
            base=native._benchmark_mixed(variant,'compute_only',ws,acs,0,1,2,'64x128x256','group_major',59)['output']
            for impl in args.implementations:
                # Inverse layout is OUTSIDE timing, solely to feed existing validated
                # prepared-core API. This is not an end-to-end integration benchmark.
                wp,wsc=converted[wf][impl];ap,asc=converted[af][impl]
                wn=wp.permute(1,0,2).contiguous().reshape(4096,2048)
                an=ap.permute(0,2,1,3).contiguous().reshape(8192,2048)
                y=native._benchmark_roof_candidate(variant,59,an,asc,wn,wsc,0,1)['output']
                assert torch.equal(y.view(torch.int32),base.view(torch.int32)),(sid,variant,impl,'gemm')
                record=dict(sample_id=sid,variant=variant,implementation=impl,
                    output_bitwise_current_best=True,mse_vs_current_best=mse(y,base),
                    mse_vs_paired_fp16=mse(y,fp),paired_reference=mf.PAIRED_BASELINE[variant])
                correctness.append(record);append('output_mse.jsonl',record)
            del fp,base,wp,wsc,ap,asc,wn,an,y
        save('summary.json',summarize(records))
        save('validation.json',dict(passed=True,records=len(records),output_checks=len(correctness),
            output_bitwise_current_best=True,gemm_performance_measured=False,conversion_only=True))
        del sources,converted,raw
    print('CONVERSION PROBE PASSED',flush=True)


if __name__=='__main__': main()


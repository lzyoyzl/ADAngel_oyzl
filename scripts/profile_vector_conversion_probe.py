#!/usr/bin/env python3
"""Single-input conversion profiling target; never substitute NCU time for Event results.

Run this under ncu, filtering conversion entry names and skipping 50 matched
warmup launches. Source quantization/provenance checks occur before the probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark_a100_mixed_trace import inspect_inputs,inspect_raw_inputs,verify_raw_prepared,source_identity
from vector_conversion_probe import VectorConversionProbe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--format',choices=['nvfp4_g128','mxfp8_e4m3_g128','hif4_g128','nvstyle_fp6_e2m3_g128'],required=True)
    p.add_argument('--policy',type=int,choices=[0,1,2],required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh profile receipt directory required')
    import torch
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.trace.prepare import _load_and_validate_raw
    from adangel.trace.storage import load_prepared,sha256_file
    torch.set_num_threads(4)
    data=Path('data/prepared/llama2_7b_prefill_o0_o4');raw_data=Path('data/raw/llama2_7b_prefill')
    manifest,mh=inspect_inputs(data)
    raw_manifest,rh=inspect_raw_inputs(raw_data,manifest,Path('configs/trace/llama2_7b_prefill.yaml'))
    entry=manifest['samples'][0];raw_entry=next(r for r in raw_manifest['samples'] if r['sample_id']==entry['sample_id'])
    if sha256_file(data/entry['file'])!=entry['sha256'] or sha256_file(raw_data/raw_entry['file'])!=raw_entry['sha256']:
        raise ValueError('source identity mismatch')
    raw=_load_and_validate_raw(raw_data/raw_entry['file'],raw_entry['layer'],raw_entry['projection'])
    original=load_prepared(data/entry['file'],device='cpu')
    verify_raw_prepared(original,(raw['activation_fp16'],raw['weight_fp16']))
    weight=mf.FORMATS[a.format][1]==4
    source=mf.quantize_source(raw['weight_fp16' if weight else 'activation_fp16'].cuda(),a.format)
    q,sc=mf.to_fixed_reference(source);rows,k=source['shape'];g=k//128
    packed=mf._pack_nibbles(q.to(torch.uint8)&15) if weight else split_int8_to_packed_int4(q)
    expected=(packed.reshape(rows,g,64).permute(1,0,2) if weight else
              packed.reshape(2,rows,g,64).permute(0,2,1,3)).contiguous()
    result=VectorConversionProbe(a.library)(source,a.policy,50,1,1)
    assert torch.equal(result['packed'],expected)
    assert torch.equal(result['scale'].contiguous().view(torch.int32),sc.contiguous().view(torch.int32))
    a.output.mkdir(parents=True)
    (a.output/'profile_receipt.json').write_text(json.dumps(dict(
        scope='single_conversion_NCU_target_not_formal_Event_result',passed=True,sample_id=entry['sample_id'],
        format=a.format,policy=a.policy,matched_launches=51,target_launch_index_zero_based=50,
        source=source_identity(source),prepared_manifest_sha256=mh,raw_manifest_sha256=rh,
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')


if __name__=='__main__':main()

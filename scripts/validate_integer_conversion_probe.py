#!/usr/bin/env python3
"""v34 finite encoding/layout validation; no production-default switch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    if args.output.exists(): p.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('SM80 required')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    checks=[]

    def move(source):
        return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in source.items()}

    def verify(source,label):
        q,scale=mf.to_fixed_reference(source)
        weight=mf.FORMATS[source['format']][1]==4
        packed=mf._pack_nibbles(q.to(torch.uint8)&15) if weight else split_int8_to_packed_int4(q)
        rows,k=source['shape'];groups=k//128
        expected=(packed.reshape(rows,groups,64).permute(1,0,2) if weight else
                  packed.reshape(2,rows,groups,64).permute(0,2,1,3)).contiguous()
        gpu=move(source)
        for impl in (0,1,2,3):
            stream=torch.cuda.Stream()
            with torch.cuda.stream(stream):
                result=native._benchmark_mixed_conversion_probe(gpu,impl,0,1,2)
            stream.synchronize()
            assert torch.equal(result['packed'].cpu(),expected),(label,impl,'payload')
            got=result['scale'].cpu().contiguous()
            assert torch.equal(got.view(torch.int32),scale.contiguous().view(torch.int32)),(label,impl,'scale')
            assert result['scale'].stride()==(1,rows)
            assert result['inner_repeats']==2 and len(result['timings_ms'])==1
            checks.append(dict(format=source['format'],case=label,shape=source['shape'],
                               implementation=impl,bitwise=True,nondefault_stream=True))

    for fmt,(kind,_,_) in mf.FORMATS.items():
        codes=[c for c in range(2*mf.SIGN_BITS[kind]) if not(kind=='e4m3' and c&127==127)]
        for micro in range(4 if fmt=='hif4_g128' else 1):
            source=mf.quantize_source(torch.ones(2,256),fmt)
            values=torch.tensor((codes*32)[:512],dtype=torch.uint8).reshape(2,256)
            source['payload']=mf._pack_nibbles(values) if kind in ('e2m1','s1p2') else values
            if fmt=='hif4_g128':
                source['micro8'].fill_(255 if micro&1 else 0)
                source['micro4'].fill_(255 if micro&2 else 0)
                source['scale']=torch.tensor([[0,180],[192,254]],dtype=torch.uint8)
            elif fmt=='mxfp8_e4m3_g128': source['scale']=torch.tensor([[0,120],[127,252]],dtype=torch.uint8)
            else:
                source['scale']=torch.tensor([[0,1],[64,126]],dtype=torch.uint8)
                source['tensor_scale']=torch.tensor([.0317],dtype=torch.float32)
            verify(source,f'all_finite_codes_micro{micro}')
        count=253 if fmt=='mxfp8_e4m3_g128' else (255 if fmt=='hif4_g128' else 127)
        source=mf.quantize_source(torch.ones(count,128),fmt)
        source['scale']=torch.arange(count,dtype=torch.uint8).reshape(count,1)
        verify(source,'all_valid_scale_codes')
        for rows in ((1,7) if args.quick else (1,3,4,7,64,129)):
            for k in ((128,384) if args.quick else (128,256,384,640,4096)):
                source=mf.quantize_source(torch.linspace(-8,8,rows*k).reshape(rows,k),fmt)
                verify(source,'tail_and_odd_groups')

    rejects=[]
    for fmt in mf.FORMATS:
        source=move(mf.quantize_source(torch.ones(1,128),fmt))
        source['scale'].fill_(255)
        for impl in (0,1,2,3):
            try: native._benchmark_mixed_conversion_probe(source,impl,0,1,2)
            except (RuntimeError,ValueError): rejects.append(dict(format=fmt,implementation=impl,rejected='invalid_scale'))
            else: raise AssertionError('invalid scale accepted')
    source=move(mf.quantize_source(torch.ones(1,128),'mxfp8_e4m3_g128'))
    for code in (127,255):
        source['payload'].fill_(code)
        for impl in (0,1,2,3):
            try: native._benchmark_mixed_conversion_probe(source,impl,0,1,2)
            except (RuntimeError,ValueError): rejects.append(dict(implementation=impl,rejected=f'nan_payload_{code}'))
            else: raise AssertionError('NaN payload accepted')
    report=dict(passed=True,checks=checks,rejected=rejects,scope='conversion_payload_and_scale_only',
        gemm_measured=False,quick=args.quick,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),rejected=len(rejects))))


if __name__=='__main__': main()

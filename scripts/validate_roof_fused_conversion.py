#!/usr/bin/env python3
"""v25: exact source decoding + fused payload layout; not performance acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true',help='small synthetic geometry for sanitizer')
    args=parser.parse_args()
    if args.output.exists(): parser.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from benchmark_a100_mixed import integer_reference
    from roof_payload_validation import verify_grouped_payload
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('SM80 required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True)
    checks=[]

    def move(source):
        return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in source.items()}

    def convert(source,label):
        q,scale=mf.to_fixed_reference(source)
        weight=mf.FORMATS[source['format']][1]==4
        packed=mf._pack_nibbles(q.to(torch.uint8)&15) if weight else split_int8_to_packed_int4(q)
        rows,k=source['shape'];groups=k//128
        expected=(packed.reshape(rows,groups,64).permute(1,0,2) if weight else
                  packed.reshape(2,rows,groups,64).permute(0,2,1,3)).contiguous()
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result=native._convert_mixed_fused_payload(move(source))
        stream.synchronize()
        assert torch.equal(result['packed'].cpu(),expected),(label,source['format'],'payload')
        assert torch.equal(result['scale'].cpu().contiguous().view(torch.int32),scale.view(torch.int32)),(label,'scale')
        assert result['scale'].stride()==(1,rows)
        checks.append(dict(type='codec',case=label,format=source['format'],shape=source['shape'],bitwise=True))

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
            convert(source,f'all_finite_codes_micro{micro}')
        count=253 if fmt=='mxfp8_e4m3_g128' else (255 if fmt=='hif4_g128' else 127)
        source=mf.quantize_source(torch.ones(count,128),fmt)
        source['scale']=torch.arange(count,dtype=torch.uint8).reshape(count,1)
        convert(source,'all_valid_scale_codes')
        for rows in (1,3,7):
            for k in (128,384,640):
                convert(mf.quantize_source(torch.linspace(-8,8,rows*k).reshape(rows,k),fmt),'tail_and_odd_groups')
    for fmt in mf.FORMATS:
        source=mf.quantize_source(torch.ones(1,128),fmt)
        source['scale'].fill_(255)
        try: native._convert_mixed_fused_payload(move(source))
        except (RuntimeError,ValueError): checks.append(dict(type='invalid_scale',format=fmt,rejected=True))
        else: raise AssertionError(('code255 accepted',fmt))

    shapes=[(64,128,256)] if args.quick else [(64,128,256),(128,256,768),(256,256,4096)]
    for m,n,k in shapes:
        for pattern in ('random','zero','alternating'):
            gen=torch.Generator().manual_seed(251025+k)
            a=(torch.randn(m,k,generator=gen)*.5).half()
            w=(torch.randn(n,k,generator=gen)*.25).half()
            if pattern=='zero': a.zero_();w.zero_()
            if pattern=='alternating':
                a[:,::2]=-32;a[:,1::2]=31;w[:,::2]=-7;w[:,1::2]=7
            ai=((torch.arange(m*k).reshape(m,k)*13)%256-128).to(torch.int8).cuda()
            mx=((torch.arange(n*k//2).reshape(n,k//2)*7)%256).byte().cuda()
            if pattern=='zero': ai.zero_();mx.zero_()
            if pattern=='alternating': ai[:,::2]=-128;ai[:,1::2]=127;mx.fill_(0x7f)
            asc=torch.linspace(.001,.03,m,device='cuda')
            wsc=((torch.arange(n*k//128,device='cuda').reshape(n,-1)*7)%13+116).byte()
            for variant in ('o3','o7','o8'):
                if variant=='o3':
                    def call(tune,mode): return native.benchmark('o3',mode,ai,asc,mx,wsc,0,2,2,'production',tune)
                else:
                    wf,af=mf.VARIANTS[variant]
                    ws,asrc=move(mf.quantize_source(w,wf)),move(mf.quantize_source(a,af))
                    def call(tune,mode): return native._benchmark_mixed(variant,mode,ws,asrc,0,2,2,'64x128x256','group_major',tune)
                for old,new in ((41,43),(42,44)):
                    baseline=call(old,'compute_only')
                    if variant=='o3':
                        natural_a=split_int8_to_packed_int4(ai)
                        table=torch.tensor([0,0,1,2,2,3,4,6,0,0,-1,-2,-2,-3,-4,-6],dtype=torch.int8,device='cuda')
                        wq=torch.stack((table[(mx&15).long()],table[(mx>>4).long()]),-1).reshape(n,k)
                        natural_w=mf._pack_nibbles(wq.to(torch.uint8)&15)
                        ref=integer_reference(ai,asc[:,None].expand(m,k//128),wq,torch.exp2(wsc.float()-127))
                    else:
                        aq,sa=mf.to_fixed_reference(asrc);wq,sw=mf.to_fixed_reference(ws)
                        natural_a=split_int8_to_packed_int4(aq);natural_w=mf._pack_nibbles(wq.to(torch.uint8)&15)
                        ref=integer_reference(aq,sa,wq,sw)
                    torch.testing.assert_close(baseline['output'],ref,rtol=1e-3,atol=1e-3)
                    for mode in ('conversion_only','compute_only','cold','steady_state'):
                        result=call(new,mode)
                        assert torch.equal(result['output'].view(torch.int32),baseline['output'].view(torch.int32))
                        verify_grouped_payload(result,new,natural_a,natural_w)
                        for operand in ('activation','weight'):
                            actual=result['converted_'+operand]
                            expected=baseline['converted_'+operand]
                            if variant=='o3': assert torch.equal(actual,expected)
                            else:
                                assert torch.equal(actual[0],expected[0])
                                assert torch.equal(actual[1].contiguous().view(torch.int32),expected[1].contiguous().view(torch.int32))
                        assert all(len(t)==2 for t in result['timings_ms'].values())
                        checks.append(dict(type='four_mode',variant=variant,tune=new,mode=mode,
                            shape=[m,n,k],pattern=pattern,bitwise=True,mse_vs_previous=0.0))
    torch.cuda.synchronize()
    report=dict(passed=True,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),checks=checks,
        scope='synthetic_fused_conversion_not_real_trace_performance')
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks)),indent=2))


if __name__=='__main__': main()

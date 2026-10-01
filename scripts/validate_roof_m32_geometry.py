#!/usr/bin/env python3
"""M32 boundary checks against padded M64 control and FP64 semantics."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_a100_roof_candidates import group_major_scales
from roof_payload_validation import verify_grouped_payload
from roof_reduction_validation import reference_fp64


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        p.error('fresh output file with existing parent required')
    import torch
    from adangel import _sm80 as native
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('SM80 required')
    torch.set_num_threads(4);torch.manual_seed(20261001)
    rows=[]

    def pack(q):
        u=q.to(torch.int16)&15
        return (u[:,::2]|(u[:,1::2]<<4)).byte().contiguous()

    for m,n,k in ((32,128,128),(32,256,384),(96,128,640),(96,256,768)):
        for pattern in ('random','extrema','zero','zero_scale'):
            a=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int16)
            w=torch.randint(-8,8,(n,k),device='cuda',dtype=torch.int16)
            if pattern=='extrema':
                a[:,::2],a[:,1::2]=-128,127
                w[:,::2],w[:,1::2]=-8,7
            if pattern=='zero': a.zero_();w.zero_()
            packed_a=torch.cat((pack(a&15),pack(a>>4)))
            packed_w=pack(w)
            def scales(size,multiplier):
                r=torch.arange(size,device='cuda')[:,None]
                g=torch.arange(k//128,device='cuda')[None,:]
                return group_major_scales((1+(r*multiplier+g*29)%113/128)*torch.exp2(((r+3*g)%7-10).float()))
            asc,wsc=scales(m,13),scales(n,17)
            if pattern=='zero_scale': asc[:,::2]=0;wsc[:,1::2]=0
            values=(packed_a,asc,packed_w,wsc)
            pm=(m+63)//64*64
            padded=torch.zeros((2,pm,k//2),device='cuda',dtype=torch.uint8)
            padded[:,:m]=packed_a.reshape(2,m,k//2)
            pas=group_major_scales(torch.zeros((pm,k//128),device='cuda'))
            pas[:m]=asc
            reference=reference_fp64('o7',values)
            control=native._benchmark_roof_candidate('o7',56,padded.reshape(2*pm,k//2),pas,packed_w,wsc,0,1)['output'][:m]
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for variant in ('o7','o8'):
                    for tune in (57,58):
                        result=native._benchmark_roof_candidate(variant,tune,*values,0,1)
                        stream.synchronize()
                        y=result['output']
                        assert y.dtype==torch.float32 and torch.isfinite(y).all()
                        assert torch.equal(y.view(torch.int32),control.view(torch.int32))
                        torch.testing.assert_close(y.double(),reference,rtol=1e-3,atol=1e-3)
                        checks=verify_grouped_payload(result,tune,packed_a,packed_w,wsc)
                        rows.append(dict(variant=variant,tune=tune,shape=[m,n,k],pattern=pattern,
                            bitwise_equal_padded_control=True,finite=True,
                            max_abs_vs_semantic_fp64=float((y.double()-reference).abs().max()),
                            semantic_tolerance_passed=True,nondefault_stream=True,**checks))
            torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    payload=dict(passed=True,scope='finite_m32_geometry_not_performance_acceptance',
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),checks=rows)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print('M32 geometry checks:',len(rows),'passed')


if __name__=='__main__': main()

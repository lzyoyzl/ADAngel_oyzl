#!/usr/bin/env python3
"""v36 exact O3 conversion, same GEMM54, four-mode and alignment contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MODES=('conversion_only','compute_only','cold','steady_state')


def validate_pair(result,base,mode,implementation,inner,repeats):
    import torch
    assert result['output'].dtype==torch.float32 and torch.isfinite(result['output']).all()
    assert torch.equal(result['output'].view(torch.int32),base['output'].view(torch.int32))
    for key in ('converted_weight','converted_activation','converted_weight_scale',
                'packed_activation_g128_major','packed_weight_g128_major'):
        assert torch.equal(result[key],base[key]),key
    meta=result['kernel']
    assert meta['roof_tune']==54 and meta['conversion_candidate']==implementation
    assert meta['gemm_tune']==54 and not meta['gemm_math_changed']
    assert result['weight_cached']==(mode not in ('cold','conversion_only'))
    assert result['activation_prepared']==(mode=='compute_only')
    assert result['total_timing']==('sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event')
    for stage,values in result['timings_ms'].items():
        assert len(values)==repeats and all(v>0 for v in values)
        assert result['stage_timing_inner_repeats'][stage]==(inner if 'conversion' in stage or mode=='conversion_only' else 1)
    assert meta['weight_conversion_kernels']==(1 if implementation else 3)
    assert meta['activation_conversion_kernels']==(1 if implementation else 2)
    if implementation:
        assert meta['payload_reorder_fused'] and meta['weight_scale_reorder_fused']
        assert meta['weight_payload_reorder_traffic_bytes']==meta['activation_payload_reorder_traffic_bytes']==0
        assert meta['natural_payload_export']=='diagnostic_inverse_layout_after_timing'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    if args.output.exists(): p.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.mixed_formats import _pack_nibbles
    from roof_reduction_validation import reference_fp64
    if torch.cuda.get_device_capability()!=(8,0): raise RuntimeError('A100 required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True);checks=[];rejected=[]
    shapes=[(64,128,256)] if args.quick else [(64,128,256),(128,256,768),(256,256,4096)]
    for m,n,k in shapes:
        for pattern in ('all_codes','zero','alternating','random'):
            torch.manual_seed(3610+k)
            a=((torch.arange(m*k,device='cuda')*13)%256-128).to(torch.int8).reshape(m,k)
            w=(torch.arange(n*k//2,device='cuda')%256).byte().reshape(n,k//2)
            asc=torch.linspace(0,.01,m,device='cuda')
            ws=((torch.arange(n*k//128,device='cuda')*7)%13+116).byte().reshape(n,k//128)
            if pattern=='zero': a.zero_();w.zero_()
            if pattern=='alternating': a[:,::2]=-128;a[:,1::2]=127;w.fill_(0x7f)
            if pattern=='random':
                a=torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8)
                w=torch.randint(0,256,(n,k//2),device='cuda',dtype=torch.uint8)
            def call(impl,mode): return native.benchmark('o3',mode,a,asc,w,ws,0,2,2,'production',54,impl)
            base=call(0,'compute_only')
            lut=torch.tensor([0,0,1,2,2,3,4,6,0,0,-1,-2,-2,-3,-4,-6],device='cuda',dtype=torch.int8)
            wq=torch.stack((lut[(w&15).long()],lut[(w>>4).long()]),-1).reshape(n,k)
            pa=split_int8_to_packed_int4(a);pw=_pack_nibbles(wq.byte()&15)
            assert torch.equal(base['converted_activation'],pa) and torch.equal(base['converted_weight'],pw)
            ref=reference_fp64('o3',(pa,asc,pw,ws))
            torch.testing.assert_close(base['output'].double(),ref,rtol=1e-3,atol=1e-3)
            for impl in (1,2):
                for mode in MODES:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream): result=call(impl,mode)
                    stream.synchronize()
                    validate_pair(result,base,mode,impl,2,2)
                    assert torch.equal(result['converted_weight_scale'],ws.T.contiguous())
                    checks.append(dict(shape=[m,n,k],pattern=pattern,implementation=impl,mode=mode,
                        output_bitwise=True,payload_bitwise=True,scale_bitwise=True,semantic_tolerance_passed=True,
                        mse_vs_previous=0.,nondefault_stream=True))
    for impl,tune in ((-1,54),(3,54),(1,-1),(2,52)):
        try: native.benchmark('o3','compute_only',a,asc,w,ws,0,1,2,'production',tune,impl)
        except RuntimeError: rejected.append(dict(case='dispatch',implementation=impl,tune=tune))
        else: raise AssertionError('invalid dispatch accepted')
    for operand in ('a','w'):
        original=a if operand=='a' else w
        odd=torch.empty(original.numel()+1,device='cuda',dtype=original.dtype)[1:].reshape(original.shape)
        odd.copy_(original)
        try: native.benchmark('o3','compute_only',odd if operand=='a' else a,asc,
                              odd if operand=='w' else w,ws,0,1,2,'production',54,2)
        except RuntimeError as e:
            assert 'alignment' in str(e);rejected.append(dict(case='unaligned',operand=operand))
        else: raise AssertionError('misaligned vector operand accepted')
    for code in (0,255):
        bad=ws.clone();bad[0,0]=code
        try: native.benchmark('o3','compute_only',a,asc,w,bad,0,1,2,'production',54,1)
        except RuntimeError: rejected.append(dict(case='scale_guard',code=code))
        else: raise AssertionError('GEMM54 guard was bypassed')
    report=dict(passed=True,checks=checks,rejected=rejected,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),rejected=len(rejected))))


if __name__=='__main__': main()

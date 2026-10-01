#!/usr/bin/env python3
"""v35 four-mode integration of exact fused conversion with unchanged GEMM59."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def validate_pair(result,base,mode,implementation,inner,repeats):
    import torch
    assert torch.equal(result['output'].view(torch.int32),base['output'].view(torch.int32))
    for operand in ('weight','activation'):
        for x,y in zip(result['converted_'+operand],base['converted_'+operand]):
            assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
        assert torch.equal(result['packed_'+operand+'_g128_major'],base['packed_'+operand+'_g128_major'])
    meta=result['kernel']
    assert meta['roof_tune']==59 and meta['conversion_candidate']==implementation
    assert not meta['fp32_reassociated']
    assert result['weight_cached']==(mode not in ('cold','conversion_only'))
    assert result['activation_prepared']==(mode=='compute_only')
    assert result['total_timing']==('sum_of_batched_stage_samples' if mode=='conversion_only' else 'single_execution_cuda_event')
    for stage,values in result['timings_ms'].items():
        assert len(values)==repeats
        assert result['stage_timing_inner_repeats'][stage]==(inner if 'conversion' in stage or mode=='conversion_only' else 1)
    if implementation:
        assert meta['conversion_kernels_per_operand']==1
        assert meta['payload_reorder_fused'] and not meta['gemm_math_changed']
        assert meta['weight_payload_reorder_traffic_bytes']==meta['activation_payload_reorder_traffic_bytes']==0
        assert meta['natural_payload_export']=='diagnostic_inverse_layout_after_timing'
    else:
        assert meta['conversion_kernels_per_operand']==2 and not meta['payload_reorder_fused']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    if args.output.exists(): p.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed import integer_reference
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True);checks=[];rejects=[]
    for m,n,k in ([(64,128,256)] if args.quick else [(64,128,256),(128,256,768),(256,256,4096)]):
        for pattern in ('random','zero','alternating'):
            torch.manual_seed(3510+k)
            a=(torch.randn(m,k,device='cuda')*.4).half();w=(torch.randn(n,k,device='cuda')*.1).half()
            if pattern=='zero': a.zero_();w.zero_()
            if pattern=='alternating': a[:,::2]=-8;a[:,1::2]=7;w[:,::2]=-7;w[:,1::2]=6
            for variant,(wf,af) in mf.VARIANTS.items():
                ws=mf.quantize_source(w,wf);acs=mf.quantize_source(a,af)
                def call(impl,mode): return native._benchmark_mixed(variant,mode,ws,acs,0,2,2,'64x128x256','group_major',59,impl)
                base=call(0,'compute_only')
                aq,asc=mf.to_fixed_reference(acs);wq,wsc=mf.to_fixed_reference(ws)
                torch.testing.assert_close(base['output'],integer_reference(aq,asc,wq,wsc),rtol=1e-3,atol=1e-3)
                for impl in (1,2,3,4):
                    for mode in ('conversion_only','compute_only','cold','steady_state'):
                        out=call(impl,mode)
                        validate_pair(out,base,mode,impl,2,2)
                        checks.append(dict(shape=[m,n,k],pattern=pattern,variant=variant,implementation=impl,
                            mode=mode,bitwise=True,semantic_tolerance_passed=True,mse_vs_previous=0.))
    for impl,tune in ((-1,59),(6,59),(4,56),(4,-1)):
        try: native._benchmark_mixed('o8','compute_only',ws,acs,0,2,2,'64x128x256','group_major',tune,impl)
        except RuntimeError: rejects.append(dict(implementation=impl,tune=tune,rejected=True))
        else: raise AssertionError('unsupported conversion dispatch accepted')
    report=dict(passed=True,checks=checks,rejected=rejects,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),rejected=len(rejects))))


if __name__=='__main__': main()

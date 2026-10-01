#!/usr/bin/env python3
"""v54 integrated fixed vector16, exact outputs and unchanged four-mode semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from validate_conversion_pipeline import validate_pair as validate_common


def validate_pair(result, base, mode, implementation, inner, repeats):
    validate_common(result, base, mode, implementation, inner, repeats)
    assert implementation in (4, 5)
    meta = result['kernel']
    assert meta['conversion_vector_elements'] == (16 if implementation == 5 else 0)
    if implementation == 5:
        assert meta['conversion_kernel_symbol'] == 'adangel_sm80_vector_fixed_conversion'
        assert meta['weight_conversion_impl'] == meta['activation_conversion_impl'] == 5
    expected = {'conversion_only': {'weight_conversion', 'activation_conversion', 'total'},
                'compute_only': {'gemm', 'total'},
                'cold': {'weight_conversion', 'activation_conversion', 'gemm', 'total'},
                'steady_state': {'activation_conversion', 'gemm', 'total'}}
    assert set(result['timings_ms']) == expected[mode]
    assert all(x > 0 for values in result['timings_ms'].values() for x in values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.output.exists(): p.error('fresh output required')
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from benchmark_a100_mixed import integer_reference
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    checks, rejects = [], []
    shapes = [(64,128,256)] if args.quick else [(64,128,256),(128,256,768),(256,256,4096)]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for m,n,k in shapes:
            for pattern in ('random', 'zero', 'alternating'):
                torch.manual_seed(5410+k)
                a=(torch.randn(m,k,device='cuda')*.4).half()
                w=(torch.randn(n,k,device='cuda')*.1).half()
                if pattern=='zero': a.zero_(); w.zero_()
                if pattern=='alternating':
                    a[:,::2]=-8; a[:,1::2]=7; w[:,::2]=-7; w[:,1::2]=6
                for variant,(wf,af) in mf.VARIANTS.items():
                    ws=mf.quantize_source(w,wf); acs=mf.quantize_source(a,af)
                    def call(impl,mode):
                        return native._benchmark_mixed(variant,mode,ws,acs,0,2,2,
                                                       '64x128x256','group_major',59,impl)
                    base=call(4,'compute_only')
                    aq,asc=mf.to_fixed_reference(acs); wq,wsc=mf.to_fixed_reference(ws)
                    torch.testing.assert_close(base['output'],integer_reference(aq,asc,wq,wsc),rtol=1e-3,atol=1e-3)
                    for impl in (4,5):
                        for mode in ('conversion_only','compute_only','cold','steady_state'):
                            result=call(impl,mode)
                            validate_pair(result,base,mode,impl,2,2)
                            checks.append(dict(shape=[m,n,k],pattern=pattern,variant=variant,
                                implementation=impl,mode=mode,bitwise=True,nondefault_stream=True,
                                semantic_tolerance_passed=True,mse_vs_previous=0.))
        for impl,tune in ((-1,59),(6,59),(5,56),(5,-1)):
            try: native._benchmark_mixed('o8','compute_only',ws,acs,0,2,2,'64x128x256','group_major',tune,impl)
            except RuntimeError: rejects.append(dict(implementation=impl,tune=tune,rejected=True))
            else: raise AssertionError('unsupported dispatch accepted')
        for operand in ('weight','activation'):
            wbad, abad=dict(ws),dict(acs)
            bad=wbad if operand=='weight' else abad
            original=bad['payload']
            flat=torch.empty(original.numel()+1,device=original.device,dtype=original.dtype)
            bad['payload']=flat[1:].view(original.shape)
            bad['payload'].copy_(original)
            assert bad['payload'].is_contiguous() and bad['payload'].data_ptr()%16
            try: native._benchmark_mixed('o8','cold',wbad,abad,0,2,2,'64x128x256','group_major',59,5)
            except RuntimeError as exc:
                assert '16-byte aligned' in str(exc)
                rejects.append(dict(operand=operand,rejected='misaligned_payload'))
            else: raise AssertionError('misaligned vector input accepted')
    stream.synchronize()
    report=dict(passed=True,checks=checks,rejected=rejects,quick=args.quick,
        scope='opt_in_vector16_four_modes_same_GEMM59',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest())
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),rejected=len(rejects))))


if __name__=='__main__': main()

#!/usr/bin/env python3
"""v106: direct full24 O7 conversion/four-mode test, identical v78 GEMM.

No small-shape performance screen. Finite synthetic tests are correctness
checks only. O8 is deliberately not repeated; its source/kernel is unchanged.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main, validate as original_validate
from probe_mx8_warp_lut_codegen import ROOT, checked, legacy_q


def timing_contract(mode,inner):
    result=eight.row_fused.timing_contract(mode,inner,1)
    result.update(comparison='v73_vs_v106_conversion_identical_v78_GEMM',
        gemm_cufunction_identical_between_policies=True,
        preparation_implementation='paired_original_row_fused_vs_MXFP8_warp_lookup',
        modified_stage='activation_payload_conversion_only',
        weight_preparation_identical=True,integer_guard_unchanged=True,
        source_format='MXFP8_E4M3_to_Q8_F_minus2',
        no_small_performance_screen=True)
    return result


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(library.parent)
        if not receipt['audit']['worth_runtime_validation']:
            raise ValueError('negative compile gate; do not launch candidate')
        super().__init__(library,baseline,candidate)
        # Both measured policies retain the exact same already-audited v78
        # CUfunction. Policy2 retains v67 solely as the numerical reference.
        self.handles[0]=self.handles[1]
        self.resources[0]=dict(self.resources[1])
        for policy in (0,1,2):
            self.resources[policy]=dict(self.resources[policy],
                preparation_implementation='MXFP8_warp_lookup' if policy==1 else 'v73_row_fused',
                GEMM_modified=False)
        fn=self.lib.roof_o78_warp_lut_benchmark
        fn.argtypes=self.lib.roof_o78_gpu_benchmark.argtypes
        fn.restype=ct.c_int
        exhaustive=self.lib.roof_mx8_warp_lut_exhaustive
        exhaustive.argtypes=[ct.c_void_p,ct.c_void_p];exhaustive.restype=ct.c_int
        self.codegen=dict(best_GEMM=self.codegen,conversion_lookup=receipt,
            runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def run(self,case,policy,mode,warmup,repeats,inner):
        import torch
        if policy not in self.handles or mode not in eight.base.MODES:
            raise ValueError('invalid conversion policy/mode')
        if np.any(case.oracle['status_flat']==2):
            raise ValueError('invalid source must not expose unwritten output')
        values=(ct.c_float*(4*repeats))()
        self.check(self.lib.roof_o78_warp_lut_benchmark(
            self.handles[policy],int(case.variant[1:]),int(policy==1),eight.base.MODES.index(mode),
            case.a_source,case.w_source,case.state_pointers,case.m,case.n,
            case.a_multiplier,case.w_multiplier,warmup,repeats,inner,
            torch.cuda.current_stream().cuda_stream,values))
        return case.state['y'],eight.base.normalize_timings(mode,np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


def validate(driver):
    import torch
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results=torch.empty((256,256,2),dtype=torch.int32,device='cuda')
        error=driver.lib.roof_mx8_warp_lut_exhaustive(results.data_ptr(),stream.cuda_stream)
        if error:raise RuntimeError('GPU exhaustive launch error '+str(error))
        stream.synchronize()
        got=results.cpu().numpy()
    codes=(np.arange(256)[:,None]+np.arange(256)[None,:])&255
    lut=np.asarray([legacy_q(c) for c in range(256)],dtype=np.int32)
    expected=lut[codes]
    if not np.array_equal(got[...,0],expected) or not np.array_equal(got[...,1],expected):
        raise ValueError('original/lookup/CPU exhaustive decoder mismatch')
    checks=original_validate(driver,variants=('o7',))
    checks['warp_lookup_exhaustive']=dict(passed=True,thread_code_combinations=65536,
        valid_E4M3FN_codes=254,invalid_127_255_not_admitted_as_sources=True,
        unchecked_legacy_numeric_slots_preserved=True,nondefault_stream=True)
    checks['scope']='GPU_exhaustive_decoder_plus_finite_small_MN_K4096_not_performance_or_full_shape_sanitizer'
    return checks


def main():
    if '--samples' in sys.argv and sys.argv[sys.argv.index('--samples')+1]!='24':
        raise SystemExit('full24 only; no small performance screen')
    if '--samples' not in sys.argv:sys.argv.extend(['--samples','24'])
    if '--full-modes' not in sys.argv:sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:sys.argv.extend(['--cubins','reports/o378_roof_v78_codegen'])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v106_codegen'),
        labels=('v73_conversion_same_v78_GEMM','v106_warp_lookup_same_v78_GEMM'),
        experiment='warp_lookup_conversion_not_GEMM',banner='MXFP8 WARP LOOKUP',
        contract=timing_contract,description=__doc__,variants=('o7',),validation_fn=validate)
    if '--validate-only' in sys.argv:return
    output=Path(sys.argv[sys.argv.index('--output')+1])
    oldpath=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old={r['sample_id']:r for r in map(json.loads,oldpath.read_text().splitlines()) if r['variant']=='o7'}
    rows=list(map(json.loads,(output/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows)!=24 or {r['sample_id'] for r in rows}!=set(old):
        raise ValueError('full24 source coverage differs from v99')
    if any(row!=old[row['sample_id']] for row in rows):
        raise ValueError('source identity differs from measured v99')
    (output/'source_identity_checked.json').write_text(json.dumps(dict(passed=True,samples=24,
        variant='o7',full_v99_source_identity_equal=True,
        old_provenance_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()

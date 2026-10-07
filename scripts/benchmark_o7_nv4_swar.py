#!/usr/bin/env python3
"""v118: full24 paired conversion/Cold test, identical original v78 GEMM.

No small performance screen. The synthetic decoder/guard tests are only
correctness gates. O3/O8 and all activation preparation remain unchanged.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main,validate as original_validate
from probe_nv4_swar_codegen import ROOT,checked


def timing_contract(mode,inner):
    result=eight.row_fused.timing_contract(mode,inner,1)
    result.update(comparison='v73_vs_v118_weight_conversion_identical_v78_GEMM',
        gemm_cufunction_identical_between_policies=True,
        preparation_implementation='paired_original_vs_packed_NVFP4_Boolean_conversion',
        modified_stage='weight_payload_decode_and_exact_square_sum_only',
        activation_preparation_identical=True,integer_guard_unchanged=True,
        source_format='NVFP4_E2M1_to_Q4_F0',no_small_performance_screen=True,
        packed_weight_and_square_scratch_checked_after_every_call=True)
    return result


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(library.parent)
        if not receipt['audit']['worth_runtime_validation']:
            raise ValueError('negative compile gate; do not launch candidate')
        super().__init__(library,baseline,candidate)
        # Keep reference2; both measured policies bind exactly the same v78.
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        for policy in (0,1,2):
            self.resources[policy]=dict(self.resources[policy],GEMM_modified=False,
                preparation_implementation='NVFP4_packed_swar' if policy==1 else 'v73_row_fused')
        fn=self.lib.roof_o78_nv4_swar_benchmark
        fn.argtypes=self.lib.roof_o78_gpu_benchmark.argtypes;fn.restype=ct.c_int
        decoder=self.lib.roof_nv4_swar_exhaustive
        decoder.argtypes=[ct.c_void_p,ct.c_void_p];decoder.restype=ct.c_int
        # Validation-only preparation also uses the candidate. Mode0 has no
        # GEMM, so invalid sources can safely exercise the original guard.
        def prepare_candidate(variant,sa,sw,state,m,n,a_mult,w_mult,stream):
            times=(ct.c_float*4)()
            return fn(self.handles[1],variant,1,0,sa,sw,state,m,n,a_mult,w_mult,0,1,2,stream,times)
        self.lib.roof_o78_gpu_prepare=prepare_candidate
        self.codegen=dict(best_GEMM=self.codegen,conversion_swar=receipt,
            runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def run(self,case,policy,mode,warmup,repeats,inner):
        import torch
        if policy not in self.handles or mode not in eight.base.MODES:
            raise ValueError('invalid packed conversion policy/mode')
        if np.any(case.oracle['status_flat']==2):
            raise ValueError('invalid source must not expose unwritten output')
        values=(ct.c_float*(4*repeats))()
        self.check(self.lib.roof_o78_nv4_swar_benchmark(
            self.handles[policy],int(case.variant[1:]),int(policy==1),eight.base.MODES.index(mode),
            case.a_source,case.w_source,case.state_pointers,case.m,case.n,
            case.a_multiplier,case.w_multiplier,warmup,repeats,inner,
            torch.cuda.current_stream().cuda_stream,values))
        # These checks are AFTER all Event intervals, never in timed kernels.
        for key in ('w','ws'):
            if not torch.equal(case.state[key].view(torch.uint8),case.expected_payload[key].view(torch.uint8)):
                raise ValueError('packed weight/scale mismatch: '+key)
        if not np.array_equal(case.state['wsq'].cpu().numpy(),case.group_squares_reference['wsq']):
            raise ValueError('exact weight square scratch mismatch')
        return case.state['y'],eight.base.normalize_timings(mode,
            np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


def validate(driver):
    import torch
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results=torch.empty((131072,4),dtype=torch.int32,device='cuda')
        error=driver.lib.roof_nv4_swar_exhaustive(results.data_ptr(),stream.cuda_stream)
        if error:raise RuntimeError('GPU exhaustive launch error '+str(error))
        stream.synchronize();got=results.cpu().numpy().view(np.uint32)
    i=np.arange(131072,dtype=np.uint32);lo=i&65535
    word=lo|(np.where(i&65536,lo^65535,lo).astype(np.uint32)<<16)
    values=np.array([0,0,1,2,2,3,4,6,0,0,-1,-2,-2,-3,-4,-6],dtype=np.int32)
    packed=np.zeros(131072,dtype=np.uint32);square=np.zeros(131072,dtype=np.uint32)
    for j in range(8):
        q=values[(word>>(4*j))&15]
        packed|=(q.astype(np.uint32)&15)<<(4*j);square+=(q*q).astype(np.uint32)
    if any(not np.array_equal(got[:,k],expected) for k,expected in enumerate((packed,packed,square,square))):
        raise ValueError('GPU scalar/SWAR/CPU packed word or square mismatch')
    checks=original_validate(driver,variants=('o7',))
    checks['packed_word_exhaustive']=dict(passed=True,words=131072,
        all_four_nibble_patterns_repeated_or_complemented=True,scalar_and_CPU_match=True,
        payload_and_square_checked=True,nondefault_stream=True)
    checks['scope']='exact_GPU_decoder_guard_and_finite_small_MN_K4096_not_performance_or_full_sanitizer'
    return checks


def main():
    if '--samples' in sys.argv and sys.argv[sys.argv.index('--samples')+1]!='24':
        raise SystemExit('full24 only; no small performance screen')
    if '--samples' not in sys.argv:sys.argv.extend(['--samples','24'])
    if '--full-modes' not in sys.argv:sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:sys.argv.extend(['--cubins','reports/o378_roof_v78_codegen'])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v118_codegen'),
        labels=('v73_conversion_same_v78_GEMM','v118_packed_swar_same_v78_GEMM'),
        experiment='NVFP4_packed_conversion_not_GEMM',banner='NVFP4 PACKED SWAR',
        contract=timing_contract,description=__doc__,variants=('o7',),validation_fn=validate)
    if '--validate-only' in sys.argv:return
    output=Path(sys.argv[sys.argv.index('--output')+1])
    oldpath=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old={r['sample_id']:r for r in map(json.loads,oldpath.read_text().splitlines()) if r['variant']=='o7'}
    rows=list(map(json.loads,(output/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows)!=24 or {r['sample_id'] for r in rows}!=set(old) or any(row!=old[row['sample_id']] for row in rows):
        raise ValueError('full24 source identity differs from v99')
    (output/'source_identity_checked.json').write_text(json.dumps(dict(passed=True,samples=24,
        variant='o7',full_v99_source_identity_equal=True,
        old_provenance_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()

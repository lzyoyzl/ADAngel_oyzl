#!/usr/bin/env python3
"""v123: exact packed FP6 conversion, full24 four-mode pairs, same v78 GEMM."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main,validate as original_validate
from probe_nv6_swar_codegen import ROOT,checked,scalar


def timing_contract(mode,inner):
    result=eight.row_fused.timing_contract(mode,inner,1)
    result.update(comparison='v73_vs_v123_activation_conversion_identical_v78_GEMM',
        gemm_cufunction_identical_between_policies=True,weight_preparation_identical=True,
        preparation_implementation='paired_original_vs_packed_FP6_DP4A_exact_norm',
        modified_stage='activation_payload_decode_and_exact_square_sum_only',
        integer_guard_unchanged=True,source_format='FP6_E2M3_to_Q6_F2',
        no_small_performance_screen=True,activation_payload_norm_checked_after_every_call=True)
    return result


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(library.parent)
        if not receipt['audit']['worth_runtime_validation']:
            raise ValueError('negative compile/resource gate; do not launch candidate')
        super().__init__(library,baseline,candidate)
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        for policy in (0,1,2):
            self.resources[policy]=dict(self.resources[policy],GEMM_modified=False,
                preparation_implementation='FP6_packed_swar' if policy==1 else 'v73_row_fused')
        fn=self.lib.roof_o78_nv6_swar_benchmark
        fn.argtypes=self.lib.roof_o78_gpu_benchmark.argtypes;fn.restype=ct.c_int
        decoder=self.lib.roof_nv6_swar_exhaustive
        decoder.argtypes=[ct.c_void_p,ct.c_void_p];decoder.restype=ct.c_int
        def prepare_candidate(variant,sa,sw,state,m,n,a_mult,w_mult,stream):
            times=(ct.c_float*4)()
            return fn(self.handles[1],variant,1,0,sa,sw,state,m,n,a_mult,w_mult,0,1,2,stream,times)
        self.lib.roof_o78_gpu_prepare=prepare_candidate
        self.codegen=dict(best_GEMM=self.codegen,conversion_swar=receipt,
            runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def run(self,case,policy,mode,warmup,repeats,inner):
        import torch
        if policy not in self.handles or mode not in eight.base.MODES:
            raise ValueError('invalid FP6 conversion policy/mode')
        if np.any(case.oracle['status_flat']==2):raise ValueError('invalid source')
        values=(ct.c_float*(4*repeats))()
        self.check(self.lib.roof_o78_nv6_swar_benchmark(
            self.handles[policy],int(case.variant[1:]),int(policy==1),eight.base.MODES.index(mode),
            case.a_source,case.w_source,case.state_pointers,case.m,case.n,
            case.a_multiplier,case.w_multiplier,warmup,repeats,inner,
            torch.cuda.current_stream().cuda_stream,values))
        # Never put host validation or reduction inside measured Event intervals.
        for key in ('a','as'):
            if not torch.equal(case.state[key].view(torch.uint8),case.expected_payload[key].view(torch.uint8)):
                raise ValueError('packed activation/scale mismatch: '+key)
        if not np.array_equal(case.state['asq'].cpu().numpy(),case.group_squares_reference['asq']):
            raise ValueError('exact activation norm scratch mismatch')
        return case.state['y'],eight.base.normalize_timings(mode,
            np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


def validate(driver):
    import torch
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results=torch.empty((131072,4),dtype=torch.int32,device='cuda')
        error=driver.lib.roof_nv6_swar_exhaustive(results.data_ptr(),stream.cuda_stream)
        if error:raise RuntimeError('GPU decoder launch error '+str(error))
        stream.synchronize();got=results.cpu().numpy().view(np.uint32)
    i=np.arange(131072,dtype=np.uint32);lo=i&65535
    word=lo|(np.where(i&65536,lo^65535,lo).astype(np.uint32)<<16)
    table=np.array([scalar(c) for c in range(256)],dtype=np.int32)
    packed=np.zeros(131072,dtype=np.uint32);square=np.zeros(131072,dtype=np.uint32)
    for j in range(4):
        q=table[(word>>(8*j))&255]
        packed|=(q.astype(np.uint32)&255)<<(8*j);square+=(q*q).astype(np.uint32)
    if any(not np.array_equal(got[:,j],expected) for j,expected in enumerate((packed,packed,square,square))):
        raise ValueError('GPU scalar/SWAR/CPU bytes or exact norm mismatch')
    checks=original_validate(driver,variants=('o8',))
    checks['packed_word_exhaustive']=dict(passed=True,words=131072,
        all_two_byte_patterns_repeated_or_complemented=True,scalar_and_CPU_match=True,
        payload_and_square_checked=True,nondefault_stream=True)
    checks['scope']='GPU_decoder_guard_small_MN_K4096_correctness_not_performance_or_full_sanitizer'
    return checks


def main():
    if '--samples' in sys.argv and sys.argv[sys.argv.index('--samples')+1]!='24':
        raise SystemExit('full24 only; no small performance screen')
    if '--samples' not in sys.argv:sys.argv.extend(['--samples','24'])
    if '--full-modes' not in sys.argv:sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:sys.argv.extend(['--cubins','reports/o378_roof_v78_codegen'])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v123_codegen_r2'),
        labels=('v73_conversion_same_v78_GEMM','v123_packed_swar_same_v78_GEMM'),
        experiment='FP6_packed_conversion_not_GEMM',banner='FP6 PACKED SWAR',
        contract=timing_contract,description=__doc__,variants=('o8',),validation_fn=validate)
    if '--validate-only' in sys.argv:return
    output=Path(sys.argv[sys.argv.index('--output')+1])
    oldpath=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    old={r['sample_id']:r for r in map(json.loads,oldpath.read_text().splitlines()) if r['variant']=='o8'}
    rows=list(map(json.loads,(output/'source_provenance.jsonl').read_text().splitlines()))
    if len(rows)!=24 or {r['sample_id'] for r in rows}!=set(old) or any(row!=old[row['sample_id']] for row in rows):
        raise ValueError('full24 source identity differs from v99')
    (output/'source_identity_checked.json').write_text(json.dumps(dict(passed=True,samples=24,
        variant='o8',full_v99_source_identity_equal=True,
        old_provenance_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()

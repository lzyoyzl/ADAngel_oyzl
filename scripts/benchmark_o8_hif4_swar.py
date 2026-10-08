#!/usr/bin/env python3
"""v138: HiF4 weight conversion pairs; both use best v123 A and v78 GEMM."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

import benchmark_o78_eight_chain_probe as eight
from benchmark_o78_coefficient_probe import main as paired_main,validate as original_validate
from probe_hif4_swar_codegen import ROOT,checked,scalar


def timing_contract(mode,inner):
    result=eight.row_fused.timing_contract(mode,inner,1)
    result.update(comparison='v73_HiF4_vs_v138_identical_v123_FP6_and_v78_GEMM',
        gemm_cufunction_identical_between_policies=True,activation_preparation_identical=True,
        activation_preparation_implementation='v123_packed_FP6',
        preparation_implementation='paired_original_vs_packed_HiF4_exact_norm',
        modified_stage='weight_payload_decode_and_exact_square_sum_only',
        integer_guard_unchanged=True,source_format='HiF4_G128_to_Q4_F0',
        no_small_performance_screen=True,payload_norm_checked_after_every_call=True)
    return result


def exhaustive_reference():
    i=np.arange(1<<20,dtype=np.uint32);lo=i&65535
    word=lo|(np.where(i&65536,lo^65535,lo).astype(np.uint32)<<16)
    e8=(i>>17)&1;e4=(i>>18)&3
    table=np.array([[[scalar(c,a,b) for c in range(16)] for b in range(2)] for a in range(2)],dtype=np.int32)
    packed=np.zeros(len(i),dtype=np.uint32);square=packed.copy()
    for j in range(8):
        q=table[e8,(e4>>(j//4))&1,(word>>(4*j))&15]
        packed|=(q.astype(np.uint32)&15)<<(4*j);square+=(q*q).astype(np.uint32)
    return word,e8,e4,packed,square


class Driver(eight.Driver):
    def __init__(self,library,baseline,candidate):
        receipt=checked(library.parent)
        if not receipt['audit']['worth_runtime_validation']:
            raise ValueError('negative compile/resource gate; no candidate launch')
        super().__init__(library,baseline,candidate)
        self.handles[0]=self.handles[1];self.resources[0]=dict(self.resources[1])
        for policy in (0,1,2):
            self.resources[policy]=dict(self.resources[policy],GEMM_modified=False,
                activation_preparation='v123_packed_FP6',
                weight_preparation='v138_packed_HiF4' if policy==1 else 'v73_row_fused_HiF4')
        fn=self.lib.roof_o78_hif4_swar_benchmark
        fn.argtypes=self.lib.roof_o78_gpu_benchmark.argtypes;fn.restype=ct.c_int
        decoder=self.lib.roof_hif4_swar_exhaustive
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
            raise ValueError('invalid conversion policy/mode')
        if np.any(case.oracle['status_flat']==2):raise ValueError('invalid source')
        values=(ct.c_float*(4*repeats))()
        self.check(self.lib.roof_o78_hif4_swar_benchmark(
            self.handles[policy],int(case.variant[1:]),int(policy==1),eight.base.MODES.index(mode),
            case.a_source,case.w_source,case.state_pointers,case.m,case.n,
            case.a_multiplier,case.w_multiplier,warmup,repeats,inner,
            torch.cuda.current_stream().cuda_stream,values))
        # Entire payloads/scales/norms checked outside measured intervals.
        for key in case.expected_payload:
            if not torch.equal(case.state[key].view(torch.uint8),case.expected_payload[key].view(torch.uint8)):
                raise ValueError('packed payload/scale mismatch: '+key)
        for key in ('asq','wsq'):
            if not np.array_equal(case.state[key].cpu().numpy(),case.group_squares_reference[key]):
                raise ValueError('exact group norm mismatch: '+key)
        return case.state['y'],eight.base.normalize_timings(mode,
            np.ctypeslib.as_array(values).reshape(4,repeats),repeats)


def validate(driver):
    import torch
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        results=torch.empty((1<<20,4),dtype=torch.int32,device='cuda')
        error=driver.lib.roof_hif4_swar_exhaustive(results.data_ptr(),stream.cuda_stream)
        if error:raise RuntimeError('GPU decoder launch error '+str(error))
        stream.synchronize();got=results.cpu().numpy().view(np.uint32)
    _,_,_,packed,square=exhaustive_reference()
    if any(not np.array_equal(got[:,j],expect) for j,expect in enumerate((packed,packed,square,square))):
        raise ValueError('GPU scalar/SWAR/CPU payload or exact norm mismatch')
    checks=original_validate(driver,variants=('o8',))
    checks['packed_word_exhaustive']=dict(passed=True,words=1<<20,
        all_four_nibble_patterns_repeated_or_complemented=True,
        all_micro8_micro4_combinations=True,scalar_and_CPU_match=True,
        payload_and_square_checked=True,nondefault_stream=True)
    checks['scope']='GPU_decoder_guard_small_MN_K4096_correctness_not_performance_or_full_sanitizer'
    return checks


def main():
    for flag,value in (('--samples','24'),('--rounds','3'),('--warmup','1000'),('--repeats','200'),('--inner','100')):
        if flag in sys.argv and sys.argv[sys.argv.index(flag)+1]!=value:
            raise SystemExit('fixed full24x3/1000/200/100 protocol; no small performance screen')
        if flag not in sys.argv:sys.argv.extend([flag,value])
    if '--full-modes' not in sys.argv:sys.argv.append('--full-modes')
    if '--cubins' not in sys.argv:sys.argv.extend(['--cubins','reports/o378_roof_v78_codegen'])
    paired_main(driver_cls=Driver,default_gpu_build=Path('reports/o378_roof_v138_codegen_r2'),
        labels=('v73_HiF4_same_v123_A_v78_GEMM','v138_HiF4_same_v123_A_v78_GEMM'),
        experiment='HiF4_packed_conversion_not_GEMM',banner='HIF4 PACKED SWAR',
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

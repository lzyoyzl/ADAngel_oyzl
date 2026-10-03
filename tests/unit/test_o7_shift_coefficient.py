"""v87 coefficient shift is exact, does not shift partial, and stays isolated."""
from pathlib import Path
import random
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o7_shift_coefficient_codegen import generated_header, eight_header, ROW_SHIFTS, NEW_COEFFICIENT, OLD_COEFFICIENT, MARKER


def test_all_legal_shift_counts_and_integer_boundaries():
    rng=random.Random(20261003)
    checked=0
    for d in range(31):
        limit=(2**31-1)//(1<<d)
        values={0,1,limit,max(0,limit-1)}|{rng.randrange(limit+1) for _ in range(512)}
        for weight_factor in values:
            coefficient=(weight_factor << (d & 31)) & 0xffffffff
            assert coefficient==weight_factor*(1<<d)<=2**31-1
            bound=min(131072,(2**31-1)//coefficient) if coefficient else 131072
            for partial in {-bound,-1,0,1,bound}:
                if abs(partial)>bound: continue
                product=partial*coefficient
                for acc in (0,-product,2**31-1-max(0,product),-2**31-min(0,product)):
                    assert -2**31<=acc<=2**31-1
                    assert -2**31<=acc+product<=2**31-1
                    assert acc+product==acc+partial*(1<<d)*weight_factor
                    checked+=1
    assert checked>100000


def test_only_coefficient_and_four_row_exponents_change():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    got=generated_header(source)
    assert got.count('shf.l.wrap.b32')==1
    assert got.count('row_shifts(ri,mi)=31-__clz(factor)')==1
    assert 'acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;' in got
    restored=got.replace('o7_shift_coefficient_experiment','o78_eight_chain_experiment').replace(
        ROW_SHIFTS,'').replace(NEW_COEFFICIENT,OLD_COEFFICIENT)
    assert restored==eight_header(source)
    assert got.count('cp.async.wait_group 0')==got.count('__syncthreads()')==1
    with pytest.raises(ValueError):
        generated_header(source.replace('// Same N64 stream and four independent MMA atoms as tune59.',
                                        '// unexpected operand stream boundary'))
    assert MARKER in got


def test_wrapper_retains_guard_and_old_fp32_fallback_no_production_dispatch():
    wrapper=(ROOT/'csrc/sm80/roof_o7_shift_coefficient_probe.cu').read_text()
    assert '#include "roof_o78_eight_chain_probe.cu"' in wrapper
    assert 'if(flag>1u) return;' in wrapper and 'if(flag==1u)' in wrapper
    assert 'O78::o3_body<64,128,128' in wrapper
    assert 'o7_shift_coefficient_experiment::body' in wrapper
    for path in (ROOT/'python/adangel/backends').glob('*.py'):
        assert 'shift_coefficient' not in path.read_text()


def test_runtime_factor_domain_and_O8_rejection_before_GPU(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from benchmark_o7_shift_coefficient import Driver, check_factor_domain
    driver=Driver.__new__(Driver)
    for method in (lambda c: driver.prepare(c), lambda c: driver.run(c,1,'compute_only',0,1,2)):
        with pytest.raises(ValueError,match='O7-only'):
            method(SimpleNamespace(variant='o8'))
    factors=np.array([[1,0],[2**30,0]],dtype=np.int32)
    oracle={'activation':{'row_status':np.array([0,1]),'factors':factors}}
    check_factor_domain(oracle)  # Fallback rows do not execute the shift path.
    for value in (0,3,-1,2**30+1):
        factors[0,0]=value
        with pytest.raises(ValueError,match='exact positive powers'):
            check_factor_domain(oracle)


def test_same_preparation_for_all_three_function_handles(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from benchmark_o7_shift_coefficient import Driver, timing_contract
    calls=[]
    def call(*args):
        calls.append(args)
        for i in range(len(args[-1])): args[-1][i]=1.0
        return 0
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        current_stream=lambda:SimpleNamespace(cuda_stream=7))))
    d=Driver.__new__(Driver); d.handles={0:11,1:12,2:13}
    d.lib=SimpleNamespace(roof_o78_gpu_benchmark=call)
    case=SimpleNamespace(variant='o7',oracle={'status_flat':np.array([0])},a_source=1,w_source=2,
        state_pointers=3,m=64,n=128,a_multiplier=4.,w_multiplier=1.,state={'y':'out'})
    for policy in (0,1,2):
        out,timings=d.run(case,policy,'compute_only',50,2,100)
        assert out=='out' and timings=={'gemm':[1.,1.],'total':[1.,1.]}
    assert [c[0] for c in calls]==[11,12,13]
    assert all(c[1:3]==(7,1) for c in calls)
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        c=timing_contract(mode,100)
        assert c['preparation_implementation']=='row_fused_conversion_factor_metadata'
        assert not c['new_preparation_or_layout'] and c['supported_variant']=='o7'
        assert c['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)


def test_variant_subset_summary_keeps_all_records_and_failures():
    sys.path.insert(0,str(ROOT/'python'))
    from benchmark_o78_coefficient_probe import summarize
    rows=[]
    for sid in ('a','b'):
        for ri in range(3):
            for p in (0,1):
                rows.append(dict(sample_id=sid,variant='o7',mode='compute_only',round=ri,candidate=p,
                    bitwise_equal_v67=True,MSE_regression_passed=True,metadata_exact=True,
                    summary=dict(median_ms=2 if p else 1,cv_percent=5),
                    stage_summaries={'gemm':dict(cv_percent=5)},mse_vs_paired_fp16=.01,max_abs_vs_v67=0))
    r=summarize(rows,('compute_only',),('o7',))
    assert len(r)==2 and r[1]['paired_speedup']==.5 and r[1]['selected_cv_failed_records']==6
    for data,variants in ((rows[:-1],('o7',)),(rows,('o7','o8')),(rows,()),(rows,('o7','o7'))):
        with pytest.raises(ValueError): summarize(data,('compute_only',),variants)


def test_static_dependency_recognition_rejects_partial_shift():
    from benchmark_o7_shift_coefficient import coefficient_dependencies
    lines=['Function : candidate']; pc=0
    for i in range(64):
        for ins in ('LDS R1, [R100]', 'LDS R2, [R101]', 'FLO.U32 R3, R1',
                    'IMMA.16864.S4.S4 R4, R8.ROW, R12.COL, RZ',
                    'SHF.L.W.U32.HI R16, RZ, R3, R2', 'IMAD R32, R4, R16, R32'):
            lines.append(f'/*{pc:04x}*/ {ins} ;'); pc+=16
    live={'loops':[dict(kind='integer',begin_pc='0x0',end_pc=hex(pc-16))]}
    result=coefficient_dependencies('\n'.join(lines),'candidate',live)
    assert len(result['coefficient_shifts'])==len(result['coefficient_imad_updates'])==64
    with pytest.raises(ValueError):
        coefficient_dependencies('\n'.join(lines).replace('RZ, R3, R2','RZ, R3, R4'),'candidate',live)

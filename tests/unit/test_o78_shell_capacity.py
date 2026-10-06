"""Compiler-independent contracts for new v110 capacity diagnostics."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_shell_capacity import generated_header,compile_gate,SYMBOLS
from run_o78_shell_capacity import summarize


def test_generated_shell_keeps_actual_math_and_only_read_only_shared():
    source=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain_generated.cuh').read_text()
    generated=generated_header(source)
    assert 'cute::_4{},cute::_2{},cute::_4{}' in generated
    assert 'SM80_16x8x64_S32U4S4S32_TN' in generated
    assert 'SM80_16x8x64_S32S4S4S32_TN' in generated
    assert 'partial(i)*=16' in generated and 'partial(vi,mi,ni)*coefficient' in generated
    assert 'sizeof(Storage)==40960' in generated
    assert generated.count('__syncthreads')==1
    assert 'cp.async' not in generated and 'prefetch(' not in generated
    assert '[group&31][cute::get<0>(coord)]' in generated
    assert '[group&31][cute::get<1>(coord)]' in generated
    assert 'base_a[' not in generated and 'base_w[' not in generated
    assert '-> auto&' not in generated
    assert 'return b20(cute::_,cute::_,cute::_);' in generated


def test_predeclared_compiler_gate_rejects_work_spill_or_occupancy_changes():
    rows={s:dict(allocated_gpr=168,loop=dict(opcode_counts={
        'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
        'LDSM.16.M88.4':0 if i==0 else 16})) for i,s in enumerate(SYMBOLS)}
    assert compile_gate(rows)['passed']
    for op in ('LDL.64','STL','BAR.SYNC.DEFER_BLOCKING','LDGSTS.E.BYPASS.128'):
        rows[SYMBOLS[1]]['loop']['opcode_counts'][op]=1
        assert not compile_gate(rows)['passed']
        del rows[SYMBOLS[1]]['loop']['opcode_counts'][op]
    rows[SYMBOLS[0]]['allocated_gpr']=176
    assert not compile_gate(rows)['passed']


def test_synthetic_reconstruction_bounds_and_host_reference_periodicity():
    import numpy as np
    for seed in (0,4):
        for row in range(10):
            a=(row+seed)%5-2;lo=a&15;hi=-1 if a<0 else 0
            assert lo+16*hi==a
            for col in range(14):
                w=(col+seed)%7-3
                for groups in (1,32,256):
                    factors=np.array([(1+((row+(g&31)+seed)&1))*(3+2*((col+((g&31)>>1)+seed)&1))
                        for g in range(groups)],dtype=np.int64)
                    value=int(np.sum(128*(lo+16*hi)*w*factors))
                    assert abs(value)<=1966080
                    assert float(np.float32(value))==value


def test_driver_residency_checksum_all_outputs_and_event_scope():
    source=(ROOT/'csrc/sm80/roof_o78_shell_capacity_driver.cpp').read_text()
    assert 'resident[mode]!=3' in source and 'regs[mode]>168' in source
    assert 'cuLaunchKernel(f[mode],N/128,M/64' in source
    assert 'std::memcmp(&got,&want,sizeof(float))' in source
    assert 'for(int m=0;m<M;++m)for(int n=0;n<N;++n)' in source
    assert 'round<3' in source and 'repeat<200' in source
    begin=source.index('for(int repeat=0;repeat<200')
    end=source.index('launch(mode,Groups,0);verify(mode,Groups,0);',begin)
    timed=source[begin:end]
    assert 'cuEventRecord(begin,stream)' in timed and 'cuEventRecord(end,stream)' in timed
    assert 'verify(' not in timed and 'cuMemAlloc(' not in timed
    assert 'selected&(1<<mode)' in timed


def test_two_shell_summary_never_claims_trace_MSE_or_relaxes_rejected_scale():
    import pytest
    rows=[dict(mode=m,round=r,symbol=SYMBOLS[m],grid=[32,64],groups=256,threads=128,
        active_ctas_per_sm=3,validation_checks=12,checksum_passed=True,shared_reserved_bytes=50688,
        raw_ms=[2.0+m*.4]*200) for r in range(3) for m in (0,1)]
    result=summarize(rows)
    assert result['modes'][0]['normalized32_groups_ms']==.25
    assert result['modes'][1]['normalized32_groups_ms']==.3
    assert not result['original_experiment_MSE_measured'] and not result['real_trace_latency_measured']
    rows[0]['validation_checks']=18
    with pytest.raises(ValueError):summarize(rows)

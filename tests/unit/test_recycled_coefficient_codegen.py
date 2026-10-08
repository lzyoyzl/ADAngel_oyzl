"""v141: bounded lifetime, algebra, and fail-closed compile gate; no GPU claim."""
import copy
from pathlib import Path
import random
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_o78_recycled_coefficient_codegen import (
    LAST_MMA, PREPARE, CONSUME, generated_header, gate)


def test_source_reuses_dead_operand_only_and_preserves_pipeline():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old=eight_header(source); new=generated_header(source)
    assert new.count('auto coefficients=cute::recast<int32_t>(b0);')==1
    assert 'cute::size(coefficients))::value==8' in new
    assert new.index('cute::gemm(LA{},p,a0(')<new.index(PREPARE)<new.index(LAST_MMA)<new.index(CONSUME)
    assert 'b0(' not in new[new.index(PREPARE):new.index('  // Reuse the 64 INT32')]
    for marker in ('cp.async.wait_group 0;', '__syncthreads();',
                   'prefetch(s,1-slot,group+1', 'cute::copy(SCopy{},bc.partition_S',
                   'partial(i)*=16;', '*reinterpret_cast<float2*>(y+offset)'):
        assert new.count(marker)==old.count(marker)
    # Eight pure products may overlap the last MMA; no barrier/opcode hack.
    assert 'asm volatile("mul' not in new
    assert '"memory"' not in PREPARE
    with pytest.raises(ValueError):
        generated_header(source.replace('  // Reuse the 64 INT32', '  // Changed frozen boundary'))


def test_bounded_window_exact_signed_arithmetic():
    rng=random.Random(141)
    for _ in range(120):
        acc=[[rng.randrange(-1_000_000,1_000_001) for _ in range(8)] for _ in range(4)]
        expected=copy.deepcopy(acc)
        for _ in range(32):
            row=[rng.randrange(1,5) for _ in range(4)]
            col=[rng.randrange(1,5) for _ in range(8)]
            partial=[[rng.randrange(-131072,131073) for _ in range(8)] for _ in range(4)]
            for ni in range(4):
                scratch=[row[r]*col[2*ni+c] for r in range(4) for c in range(2)]
                assert len(scratch)==8
                for r in range(4):
                    for c in range(2):
                        n=2*ni+c
                        acc[r][n]+=partial[r][n]*scratch[2*r+c]
            for r in range(4):
                for n in range(8):
                    expected[r][n]+=(partial[r][n]*row[r])*col[n]
                    assert -(1<<31)<=expected[r][n]<(1<<31)
            assert acc==expected


def fixtures():
    loop=dict(kind='integer',static_instructions=383,opcode_counts={
        'IMMA.16864.S4.S4':32, 'IMMA.16864.U4.S4':32,
        'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':10,'LDS':8})
    entry=dict(allocated_gpr=168,loops=[loop])
    dep=dict(counts=dict(coefficient_based_accumulator_updates=64,
                         independent_coefficient_multiplies=64,partial_first_multiplies=0),
             peak_started_not_finished_chains=8)
    return entry,copy.deepcopy(entry),dep


def test_predeclared_gate_requires_dependency_and_cost():
    old,new,dep=fixtures()
    assert gate(old,new,dep)['passed']
    for field,value in [('coefficient_based_accumulator_updates',63),
                        ('independent_coefficient_multiplies',63),('partial_first_multiplies',1)]:
        bad=copy.deepcopy(dep);bad['counts'][field]=value
        assert not gate(old,new,bad)['passed']
    bad=copy.deepcopy(dep);bad['peak_started_not_finished_chains']=6
    assert not gate(old,new,bad)['passed']
    for op in ('LDL','STL','LDS','LDSM.16.M88.4','LDGSTS.E.BYPASS.128'):
        bad=copy.deepcopy(new)
        bad['loops'][0]['opcode_counts'][op]=bad['loops'][0]['opcode_counts'].get(op,0)+1
        assert not gate(old,bad,dep)['passed']
    bad=copy.deepcopy(new);bad['allocated_gpr']=169
    assert not gate(old,bad,dep)['passed']
    bad=copy.deepcopy(new);bad['loops'][0]['static_instructions']=384
    assert not gate(old,bad,dep)['passed']


def test_wrapper_preserves_fallback_and_exposes_no_python_binding():
    text=(ROOT/'csrc/sm80/roof_o78_recycled_coefficient_probe.cu').read_text()
    assert 'if(flag>1u) return;' in text and 'if(flag==1u)' in text
    assert 'O78::o3_body<' in text and '__launch_bounds__(128,3)' in text
    assert 'PYBIND' not in text

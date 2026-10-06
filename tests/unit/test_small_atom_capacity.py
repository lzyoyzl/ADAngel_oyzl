import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('small_atom',ROOT/'scripts/probe_small_atom_capacity.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def rows(small_ms):
    return [dict(mode=i,shape=([16,8,64],[8,8,32])[i],blocks=2048,threads=128,groups=256,
        mma_per_warp_group=(64,256)[i],logical_partial_registers=32,source_chains=(8,16)[i],
        active_ctas_per_sm=3,warmup=50,validation_checks=24,checksum_passed=True,
        raw_ms=[2.0 if i==0 else small_ms]*200) for i in range(2)]


def test_equal_physical_work_despite_fourfold_instruction_count():
    r=rows(2.0)
    assert module.physical_work(r[0])==module.physical_work(r[1])==2199023255552


def test_capacity_gate_does_not_claim_case_speedup_or_MSE():
    r=module.summarize(rows(2.1))
    assert not r['full_GEMM_worth_implementing']
    assert not r['production_default_changed'] and not r['new_experiment_MSE_measured']
    assert r['no_filtering']
    assert module.summarize(rows(1.7))['full_GEMM_worth_implementing']


def test_source_observes_all_registers_and_keeps_native_types():
    text=(ROOT/'csrc/sm80/mma_small_atom_capacity_probe.cu').read_text()
    for shape in ('m8n8k32','m16n8k64'):
        for types in ('s32.u4.s4.s32','s32.s4.s4.s32'):
            assert shape+'.row.col.'+types in text
    assert 'checksum+=p[i][j]' in text and 'for(int j=0;j<Width;++j)' in text
    assert 'tid+i*Width+j' in text and '1024ll*t+15872' in text

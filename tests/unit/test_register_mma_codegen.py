"""v119 source/compile gates; not correctness or performance acceptance."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_register_mma_codegen as probe
from probe_o78_eight_chain_codegen import generated_header


def arch_source():
    return (ROOT/'third_party/cutlass-src/include/cute/arch/mma_sm80.hpp').read_text()


def test_atoms_only_change_names_and_volatility_not_instructions_or_constraints():
    old=probe.original_operations(arch_source())
    new=probe.generated_atoms(arch_source())
    assert new.count('asm(')==2 and 'asm volatile(' not in new
    for name,replacement in probe.OPERATIONS.items():
        assert old[name].replace(name,replacement).replace('asm volatile(','asm(') in new
        assert 'MMA_Traits<::'+replacement+'> : MMA_Traits<'+name+'>' in new
    assert 'mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32' in new
    assert 'mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32' in new
    assert 'satfinite' not in new and '"memory"' not in new


def test_body_is_same_v78_except_atom_and_namespace():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    original=generated_header(source)
    changed=probe.generated_body(source).replace('o78_register_mma_experiment','o78_eight_chain_experiment')
    for old,new in probe.OPERATIONS.items():
        changed=changed.replace('cute::MMA_Atom<::'+new+'>','cute::MMA_Atom<cute::'+old+'>')
    assert changed==original
    assert '__syncthreads();' in changed and 'cp.async.wait_group 0;' in changed


def fake_live(work=383,registers=168,local=0):
    return dict(allocated_gpr=registers,loops=[dict(kind='integer',static_instructions=work,
        opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16,
            'LDGSTS.E.BYPASS.128':10,'BAR.SYNC':1,'LDL':local})])


def test_gate_does_not_accept_identical_or_tiny_changed_code():
    assert not probe.gate(fake_live(),fake_live())['passed']
    assert not probe.gate(fake_live(),fake_live(work=377))['passed']
    assert probe.gate(fake_live(),fake_live(work=363))['passed']
    assert probe.gate(fake_live(),fake_live(registers=128))['passed']
    assert not probe.gate(fake_live(),fake_live(work=350,local=1))['passed']
    assert not probe.gate(fake_live(),fake_live(work=410,registers=128))['passed']


def test_scope_and_wrapper_keep_status_guard_original_fallback_and_defaults():
    wrapper=(ROOT/'csrc/sm80/roof_o78_register_mma_probe.cu').read_text()
    assert '#include "roof_o78_eight_chain_probe.cu"' in wrapper
    assert 'if(flag>1u) return;' in wrapper and 'if(flag==1u)' in wrapper
    assert 'O78::o3_body<64,128,128' in wrapper and '__launch_bounds__(128,3)' in wrapper
    code=Path(probe.__file__).read_text()
    assert 'production_default_changed=False' in code and 'changed_math=False' in code
    assert "'-arch=sm_80'" in code and "'--life-range-mode','count'" in code
    assert 'no candidate launch' in code and 'O3 migration' in code

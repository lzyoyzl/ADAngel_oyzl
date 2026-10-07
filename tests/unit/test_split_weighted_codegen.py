"""CPU algebra/source checks; do not assert CUDA runtime correctness."""
import importlib.util
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('split_weighted',ROOT/'scripts/probe_split_weighted_codegen.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
MASK=(1<<32)-1


def test_modular_split_matches_exact_guarded_sum():
    def check(acc,low,high,factor):
        expected=acc+(low+16*high)*factor
        bits=((acc+low*factor)+(high*((16*factor)&MASK)))&MASK
        assert bits==expected&MASK
        if -(1<<31)<=expected<(1<<31):
            signed=bits if bits<(1<<31) else bits-(1<<32)
            assert signed==expected
    # Include deliberately cancelling terms, overflowing16*factor, signed
    # extremes and zero. Do not silently require the individual terms to fit.
    for factor in (0,1,2,1<<20,1<<27,1<<30,2147483647):
        for acc in (-2147483648,-1,0,1,2147483647):
            for high in (-8192,-1,0,1,8192):
                for low in (-16*high,-16*high+1,0,15360):
                    check(acc,low,high,factor)
    rng=random.Random(125)
    for _ in range(100000):
        check(rng.randrange(-(1<<31),1<<31),rng.randrange(-15360,15361),
              rng.randrange(-8192,8193),rng.randrange(1<<31))


def test_generated_preserves_guard_supply_epilogue_and_semantics():
    text=mod.generated_header()
    old=mod.generated_headers('o3')[0]
    assert text[:text.index(mod.BODY)]==old[:old.index(mod.BEGIN)].replace(
        'o3_grouped_cta_experiment','o3_split_weighted_experiment')
    assert text[text.index(mod.END):]==old[old.index(mod.END):].replace(
        'o3_grouped_cta_experiment','o3_split_weighted_experiment')
    assert text.count('mad.lo.s32')==2
    assert 'uint32_t(coefficient)*16u' in text
    assert 'partial(i)*=16' not in text
    assert 'auto ph=cute::make_fragment_like(pl)' in text
    wrapper=(ROOT/'csrc/sm80/roof_o3_split_weighted_probe.cu').read_text()
    assert 'if(flag&6u) return;' in wrapper and 'if(flag&1u)' in wrapper
    assert 'o3_grouped_fallback::o3_body' in wrapper


def test_gate_rejects_resource_and_work_regression():
    counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
            'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':9,'BAR.SYNC.DEFER_BLOCKING':1}
    def record(static=323,regs=168,extra=None):
        return dict(allocated_gpr=regs,loops=[dict(kind='integer',static_instructions=static,
                         opcode_counts=counts|(extra or {}))])
    chain=dict(chain_count=32,mma_per_chain=2,peak_started_not_finished=8)
    assert mod.cost_gate(record(),record(340),chain)['passed']
    assert not mod.cost_gate(record(),record(350),chain)['passed']
    assert not mod.cost_gate(record(),record(regs=176),chain)['passed']
    assert not mod.cost_gate(record(),record(extra={'LDL':1}),chain)['passed']
    assert not mod.cost_gate(record(),record(),chain|{'peak_started_not_finished':4})['passed']

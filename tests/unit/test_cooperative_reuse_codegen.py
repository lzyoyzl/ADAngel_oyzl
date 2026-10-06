"""One proportional12-warp prototype, not duplicate larger per-thread tiles."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_cooperative_reuse_codegen import generated_header,work_model,gate


def test_new_cooperation_keeps_math_budget_not_v57_v83_v98():
    source=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain_generated.cuh').read_text()
    new=generated_header(source)
    assert 'cute::Shape<cute::_4,cute::_3,cute::_1>' in new
    assert 'cute::Tile<cute::_128,cute::_192,cute::_64>' in new
    assert 'cute::Tile<cute::_128,cute::_96,cute::_64>' in new
    assert 'static_assert(decltype(cute::size(acc))::value==64)' in new
    assert 'cute::_4{},cute::_2{},cute::_4{}' in new
    assert 'partial(i)*=16' in new and 'partial(vi,mi,ni)*coefficient' in new
    assert 'constexpr int Groups=32' in new and 'slot=group%2' in new
    assert new.count('cute::gemm(HA{}')==2 and new.count('cute::gemm(LA{}')==2
    assert 'guarded_copy16' in new and 'column<n?column:0u' in new
    assert 'global_column<n?base_w[global_column]:0.0f' in new
    assert 'sizeof(Storage)==59904' in new


def test_work_model_accounts_for_padding_and_only_valid_global_reads():
    r=work_model()
    assert r['old_ctas']==2048 and r['new_ctas']==704
    assert r['padded_mma_ratio']==1.03125
    assert r['old_resident_warps']==r['intended_resident_warps']==12
    assert .59<r['input_bytes_ratio']<.61
    assert r['acc_per_thread']==64 and r['partial_per_thread']==32


def test_compile_gate_requires_work_reduction_not_just_larger_tile():
    ops={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16}
    old=dict(allocated_gpr=168,loop=dict(static_instructions=383,opcode_counts=ops))
    new=dict(allocated_gpr=168,loop=dict(static_instructions=340,opcode_counts=dict(ops)))
    assert gate(old,new)['passed']
    new['loop']['static_instructions']=360
    assert not gate(old,new)['passed']
    new['loop']['static_instructions']=340;new['loop']['opcode_counts']['LDL']=3
    assert not gate(old,new)['passed']


def test_old_guard_covers_every_valid_element_with_new_tile_boundaries():
    # For each valid output, original64x128 proof is one of the up to4 read.
    for ty in range(32):
        for tx in range(22):
            first=tx*192//128
            old={(ty*2+dy,first+dx) for dy in range(2) for dx in range(2) if first+dx<32}
            for row in range(ty*128,(ty+1)*128):
                for col in range(tx*192,min((tx+1)*192,4096)):
                    assert (row//64,col//128) in old
    source=(ROOT/'csrc/sm80/roof_o78_cooperative_reuse_probe.cu').read_text()
    assert '__launch_bounds__(384,1)' in source and 'if(flag!=0u)return;' in source
    assert 'MUST NOT be benchmarked/exposed' in source

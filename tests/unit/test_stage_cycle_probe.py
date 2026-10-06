from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_stage_cycle_codegen import CONFIG, cycle_groups, generated_header, worth_runtime, cycle_liveness


@pytest.mark.parametrize('kind',('o3','o78'))
def test_same_pipeline_order_no_slot_reuse_before_reader_completion(kind):
    stages = CONFIG[kind]['stages']
    resident = {i:i for i in range(stages-1)}
    schedule = cycle_groups(stages)
    assert [g for g,_,_ in schedule] == list(range(32))
    for group,slot,future in schedule:
        assert resident[slot] == group
        if future is not None:
            nxt,next_slot = future
            assert next_slot != slot and nxt < 32
            assert next_slot not in resident or resident[next_slot] < group
            resident[next_slot] = nxt
    text = generated_header(kind)
    assert 'slot=group%' not in text
    assert text.count('auto process_group=') == 1
    assert '#pragma unroll 1' in text
    assert 'cycle<10' in text if kind == 'o3' else 'cycle<15' in text
    assert 'process_group(cute::Int<0>{},30,cute::Int<' + ('0' if kind == 'o3' else '1') + '>{});' in text
    assert 'process_group(cute::Int<1>{},31,cute::Int<0>{});' in text
    assert 'Eight independent chains' in text and 'Same32 logical partial registers' in text
    assert '__syncthreads();' in text and 'cp.async.commit_group' in text
    assert 'SM80_16x8x64_S32U4S4S32_TN' in text
    assert 'SM80_16x8x64_S32S4S4S32_TN' in text
    assert 'partial(vi,mi,ni)*coefficient' in text
    if kind == 'o3':
        assert 'const int coefficient=s.factor[slot][cute::get<1>(coord)];' in text
        assert 'roof_grouped_cta::tile()' in text
    else:
        assert 's.activation_factors[slot][cute::get<0>(coord)]*' in text
        assert 's.weight_factors[slot][cute::get<1>(coord)]' in text


def test_compile_gate_requires_meaningful_net_improvement():
    old = dict(allocated_gpr=168, loops=[dict(kind='integer',static_instructions=383)])
    counts = {'IMMA.16864.S4.S4':64,'IMMA.16864.U4.S4':64,'LDSM.16.M88.4':32}
    new = dict(allocated_gpr=168, loops=[dict(opcode_counts=counts,normalized_instructions_per_group=360)])
    assert worth_runtime(old,new,2)
    assert not worth_runtime(old,dict(new,allocated_gpr=176),2)
    assert not worth_runtime(old,dict(new,loops=[dict(new['loops'][0],normalized_instructions_per_group=380)]),2)
    assert not worth_runtime(old,dict(new,loops=[dict(new['loops'][0],opcode_counts={**counts,'LDL.64':1})]),2)
    assert not worth_runtime(old,new,3)


def test_public_defaults_and_guard_unchanged():
    for kind,cfg in CONFIG.items():
        text = (ROOT/f'csrc/sm80/roof_{kind}_stage_cycle_probe.cu').read_text()
        assert cfg['symbol'] in text
        assert '__launch_bounds__(128,3)' in text
        assert 'if(flag' in text and 'else' in text
        assert 'register_partial' not in text


def test_unsupported_ring_rejected():
    with pytest.raises(ValueError):
        cycle_groups(4)


def test_normalized_cycle_parser_does_not_mislabel_extra_MMA_as_speedup():
    name = 'adangel_test_cycle'
    lines = ['//-------------------- .text.'+name+' ', '// SHI_REGISTERS=168', '.L_cycle:']
    for i in range(128):
        op = 'IMMA.16864.S4.S4' if i < 64 else 'IMMA.16864.U4.S4'
        lines.append(f' /*{i*16:x}*/ {op} R4, R8, R12, R4; // | 160 |')
    lines.append(' /*800*/ BRA `(.L_cycle); // | 150 |')
    result = cycle_liveness('\n'.join(lines),name,2)
    assert result['allocated_gpr'] == 168
    assert result['loops'][0]['groups_per_iteration'] == 2
    assert result['loops'][0]['normalized_instructions_per_group'] == 64.5
    with pytest.raises(ValueError):
        cycle_liveness('\n'.join(lines),name,3)

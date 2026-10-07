"""Same-group load-only edit, exact source recovery and baseline SASS proof."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


def test_only_load_placement_changes():
    from probe_o3_late_low_codegen import generated_header,OLD,NEW,INSERT
    from probe_grouped_cta_codegen import generated_headers
    old=generated_headers('o3')[0];new=generated_header()
    assert new.replace('o3_late_low_experiment','o3_grouped_cta_experiment').replace(
        NEW,OLD).replace('    load_high(slot);','    load_a(slot,a00,a01,h00,h01);').replace(INSERT,'')==old
    assert new.count('load_low(slot);')==1
    assert new.index('h01(cute::_,mi')<new.index('load_low(slot);')<new.index('partial(i)*=16;')
    assert new.count('__syncthreads()')==old.count('__syncthreads()')
    assert new.count('cute::copy(')==old.count('cute::copy(')
    base=(ROOT/'csrc/sm80/roof_o3_crossk_load_probe.cu').read_text()
    wrapper=(ROOT/'csrc/sm80/roof_o3_late_low_probe.cu').read_text()
    assert wrapper[wrapper.index('  const uint32_t flag='):].replace('late_low','crossk_load')==base[base.index('  const uint32_t flag='):]


def test_baseline_has_two_early_and_two_delayed_low_loads():
    from probe_o3_late_low_codegen import operand_load_order,CONTROL,LIMITS
    d=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen'
    r=json.loads((d/'codegen.json').read_text())
    order=operand_load_order((d/'o3_grouped_cta.sass').read_text(),CONTROL,r['liveness'][CONTROL])
    assert [x['mma_before'] for x in order['U4']]==[0,0,12,13]
    assert LIMITS['min_first_low_mma_ordinal']==8
    assert LIMITS['max_work_ratio']==1.02 and LIMITS['max_hot_local']==0

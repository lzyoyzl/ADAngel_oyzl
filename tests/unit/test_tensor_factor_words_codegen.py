"""v130 fixes exact input-register construction, never coefficient precision."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


def test_same_payload_guard_and_coordinates_only_register_construction_differs():
    from probe_o78_tensor_factor_words_codegen import generated_header,NEW,OLD,SETUP
    from probe_o78_eight_chain_codegen import generated_header as eight
    old=eight((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    new=generated_header()
    assert new.replace('o78_tensor_factor_words_experiment','o78_eight_chain_experiment').replace(
        SETUP,'').replace(NEW,OLD)==old
    assert 'make_tensor<uint8_t>' not in NEW and 'make_tensor<uint32_t>' in NEW
    assert 'SM80_16x8x16_S32U8U8S32_TN::fma' in NEW
    original=(ROOT/'csrc/sm80/roof_o78_tensor_factor_probe.cu').read_text().split('#include',1)[1]
    current=(ROOT/'csrc/sm80/roof_o78_tensor_factor_words_probe.cu').read_text()
    # Startup range guard and fallback body are unchanged.
    guard=original[original.index('  bool too_wide='):]
    current=current[current.index('  bool too_wide='):]
    assert current.replace('tensor_factor_words','tensor_factor')==guard


def test_all_u8_words_have_only_low_byte_and_coefficient_is_exact():
    for a in range(256):
        assert a.to_bytes(4,'little')==bytes((a,0,0,0))
        for w in range(256):
            assert sum(x*y for x,y in zip(a.to_bytes(4,'little'),w.to_bytes(4,'little')))==a*w
    from probe_o78_tensor_factor_words_codegen import LIMITS
    assert LIMITS['max_work_ratio']==1.01 and LIMITS['max_hot_local']==1
    assert LIMITS['min_imad_reduction']==.20

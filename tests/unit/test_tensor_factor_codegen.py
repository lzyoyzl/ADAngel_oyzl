"""CPU algebra/source/gate tests, not GPU safety or measured performance."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_tensor_factor_codegen import generated_header,OLD,SETUP,cost_gate
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_outer_product_exact_all_u8_pairs_and_range_guard():
    for a in range(256):
        for w in range(256):
            av=[a]+[0]*15;wv=[w]+[0]*15
            value=sum(x*y for x,y in zip(av,wv))
            assert value==a*w and value<=65025
    for value in (-2147483648,-1,256,512,3584,2147483647):
        assert (value&0xffffffff)>255


def test_only_coefficient_path_changes_and_safe_fallback_exists():
    old=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    new=generated_header()
    normalized=new.replace('o78_tensor_factor_experiment','o78_eight_chain_experiment').replace(SETUP,'')
    from probe_o78_tensor_factor_codegen import NEW
    assert normalized.replace(NEW,OLD)==old
    assert 'SM80_16x8x16_S32U8U8S32_TN' in new
    assert 'SM80_16x8x64_S32U4S4S32_TN' in new and 'SM80_16x8x64_S32S4S4S32_TN' in new
    assert 'cute::get<1>(p)==0?' in new
    wrapper=(ROOT/'csrc/sm80/roof_o78_tensor_factor_probe.cu').read_text()
    assert 'if(flag>1u) return;' in wrapper and 'if(flag==1u)' in wrapper
    assert '__syncthreads_or(too_wide)' in wrapper
    assert wrapper.count('>255u')==2
    assert 'o78_eight_chain_experiment::body' in wrapper


def test_gate_requires_reduced_integer_work_not_just_more_tensor_mma():
    counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
        'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':10,'BAR.SYNC.DEFER_BLOCKING':1}
    old=dict(allocated_gpr=168,loops=[dict(kind='integer',static_instructions=383,
        opcode_counts=counts|{'IMAD':158})])
    def candidate(static=380,imad=90,regs=168,extra=None):
        return dict(allocated_gpr=regs,loops=[dict(kind='tensor_factor_integer',
            static_instructions=static,opcode_counts=counts|{'IMAD':imad,'IMMA.16816.U8.U8':16}|(extra or {}))])
    assert cost_gate(old,candidate())['passed']
    assert not cost_gate(old,candidate(static=390))['passed']
    assert not cost_gate(old,candidate(imad=140))['passed']
    assert not cost_gate(old,candidate(regs=176))['passed']
    assert not cost_gate(old,candidate(extra={'LDL':2}))['passed']

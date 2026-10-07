from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o3_dp2a_codegen import generated_header,pack_metadata_reference,gate


def test_g128_payload_geometry_reuse_and_epilogue_unchanged():
    p=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta_generated.cuh'
    text=generated_header(p.read_text())
    assert 'sizeof(Storage)==50688' in text and 'const int groups=32' in text
    assert 'factors+(33+group)*n' in text
    assert 'reinterpret_cast<const uint32_t*>(ws)[32*n+' in text
    assert text[text.index('  auto final_value='):]==p.read_text()[p.read_text().index('  auto final_value='):].replace('o3_grouped_cta_experiment','o3_dp2a_experiment')
    assert 'dp2a.lo.s32.u32' in text and 'prmt.b32' in text
    assert 'make_shape(cute::_4{},cute::_4{})' in text
    assert text.count('cute::gemm(LA{}')==2 and text.count('cute::gemm(HA{}')==2
    assert 'acc(vi,mi,full_ni)=value' in text
    assert 'partial(i)*=16' not in text


def test_metadata_pack_preserves_original_anchor_and_rejects_wide_factor():
    meta=np.ones((33,256),dtype=np.int32);meta[32]=120
    meta[:32,0]=8;meta[7,150]=1<<25
    joined=pack_metadata_reference(meta)
    assert np.array_equal(joined[:33*256],meta.ravel())
    packed=joined[33*256:65*256].reshape(32,256)
    assert (packed[:,0]==8*4097).all() and packed[7,150]==0
    assert joined[-2:].tolist()==[1,0]


def test_no_change_to_original_guard_and_uniform_branch_per_CTA():
    text=(ROOT/'csrc/sm80/roof_o3_dp2a_probe.cu').read_text()
    assert 'status[column_tile]' in text and 'if(flag&6u)return' in text
    assert 'o3_grouped_fallback::o3_body' in text and 'o3_grouped_cta_experiment::body' in text
    assert 'joined[65*n+column_tile]' in text
    assert '__syncthreads_count(!safe)' in text
    assert 'fits?value*4097:0' in text


def test_compile_gate_fails_for_more_local_work_or_MMA_change():
    ops={'IDP.2A.LO.S16.U8':64,'IMMA.16864.U4.S4':32,'IMMA.16864.S4.S4':32,'LDSM.16.M88.4':16}
    old={'loops':[dict(kind='integer',static_instructions=323,opcode_counts={'LDL':4})]}
    new=dict(allocated_gpr=168,loop=dict(static_instructions=330,opcode_counts=ops.copy()))
    assert gate(old,new)['passed']
    new['loop']['opcode_counts']['LDL']=5
    assert not gate(old,new)['passed']
    new['loop']['opcode_counts'].pop('LDL');new['loop']['opcode_counts']['IMMA.16864.S4.S4']=31
    assert not gate(old,new)['passed']


def test_shape_contract_rejects_short_metadata():
    with pytest.raises(ValueError):pack_metadata_reference(np.ones((32,128),dtype=np.int32))

"""Compile-prototype contracts, not GPU performance/accuracy evidence."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_direct_fragment_codegen import header,cost_model,compile_gate


def test_new_global_feed_not_repeated_shared_layout_or_atom_stream():
    source=ROOT/'docs/evidence/a100_o378_roof_v94/reports/o378_roof_v94_codegen/o78_operand_stream_generated.cuh'
    text=header(source.read_text())
    assert 'ld.global.ca.v4.b32' in text
    assert 'cp.async' not in text and 'cute::copy(' not in text
    assert text.count('__syncthreads()')==1
    assert 'Storage<int>)==24576' in text
    assert 'source' not in text[text.index('for(int group='):].split('  // Reuse')[0] # No new quantizer.
    assert 'partial(vi,ni)*coefficient' in text and 'partial(i)*=16' in text
    assert 'make_shape(cute::_4{},cute::_4{})' in text
    assert 'auto al=cute::recast<cutlass::uint4b_t>(ar)' in text
    assert 'words(0)=v.x;words(1)=v.y;words(2)=v.z;words(3)=v.w' in text
    assert 's.activation_factors[group]' in text and 's.weight_factors[group]' in text
    assert 'if constexpr(Integer)' in text and '__fmaf_rn(float(partial(vi,ni)),scale' in text


def test_padding_unchanged_but_logical_repeated_reads_and_pack_cost_counted():
    r=cost_model()
    assert r['cta_tile']==[64,128,128] and r['threads']==128
    assert r['acc_per_thread']==64 and r['partial_per_thread']==16
    assert r['payload_vector_reads_per_warp']==24
    assert r['new_payload_read_bytes']==3221225472
    assert r['new_metadata_read_bytes']==50331648
    assert 2.90<r['new_logical_read_ratio']<2.92
    assert r['extra_packed_allocation_4096_bytes']==25165824
    assert r['online_pack_read_write_bytes']==50331648


def test_compile_gate_demands_four_CTA_no_integer_spill_and_less_work():
    ops={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDG.E.128':24,'LDS':10}
    old=dict(allocated_gpr=168,loop=dict(static_instructions=383,opcode_counts=ops))
    new=dict(allocated_gpr=128,loop=dict(static_instructions=340,opcode_counts=dict(ops)))
    assert compile_gate(old,new)['passed']
    new['allocated_gpr']=129
    assert not compile_gate(old,new)['passed']
    new['allocated_gpr']=128;new['loop']['opcode_counts']['LDL']=1
    assert not compile_gate(old,new)['passed']
    new['loop']['opcode_counts'].pop('LDL');new['loop']['opcode_counts']['BAR.SYNC']=1
    assert not compile_gate(old,new)['passed']


def test_guard_retained_and_both_paths_use_exact_packed_operands():
    source=(ROOT/'csrc/sm80/roof_o78_direct_fragment_probe.cu').read_text()
    assert '__launch_bounds__(128,4)' in source
    assert 'if(flag>1u)return' in source
    assert 'body<false>(pa,pw,as,ws' in source and 'body<true>(pa,pw,af,wf' in source

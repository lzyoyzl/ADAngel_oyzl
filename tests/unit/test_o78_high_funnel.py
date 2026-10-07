"""CPU/source and fail-closed static checks for the sole v117 candidate."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o78_high_funnel_codegen import (CONTROL,LIMITS,NEW,ROOT,SYMBOL,cost_gate,
                                         generated_header,high_reconstruction)
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_exact_funnel_math_full_partial_and_all_legal_thread_ids():
    # Factorization proves the entire Cartesian domain, without sampling.
    tids=np.arange(1024,dtype=np.uint32)
    assert np.all((tids>>28)==0)
    p=np.arange(-8192,8193,dtype=np.int32)
    bits=p.astype(np.uint32)
    assert np.array_equal((bits<<4).view(np.int32),p*16)
    assert np.array_equal(((bits<<4)|(np.uint32(1023)>>28)).view(np.int32),p*16)


def test_source_only_changes_fixed_x16_not_coefficient_or_pipeline():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    old=eight_header(source); new=generated_header(source)
    assert NEW in new and new.count('asm("shf.l.wrap.b32')==1
    assert new.replace(NEW,'      o1_static_for<0,32>([&](auto i) { partial(i)*=16; });').replace(
        'o78_high_funnel_experiment','o78_eight_chain_experiment')==old
    assert 'threadIdx.x' in NEW and 'static_cast<unsigned>(partial(i))' in NEW
    wrapper=(ROOT/'csrc/sm80/roof_o78_high_funnel_probe.cu').read_text()
    assert '#include "roof_o78_eight_chain_probe.cu"' in wrapper
    assert '__launch_bounds__(128,3)' in wrapper and 'if(flag>1u) return;' in wrapper
    assert 'if(flag==1u)' in wrapper and 'O78::o3_body<64,128,128' in wrapper
    for f in (ROOT/'python/adangel/backends').glob('*.py'):
        assert 'high_funnel' not in f.read_text()


def fixture_sass(shift='SHF.L.W.U32.HI'):
    lines=['Function : candidate']; pc=0
    for _ in range(16):
        seq=['IMMA.16864.S4.S4 R4, R8.ROW, R12.COL, RZ',
             'IMMA.16864.S4.S4 R4, R8.ROW, R12.COL, R4']
        for i in range(4):
            args=(f'R{4+i}, 0x10, RZ' if shift=='IMAD.SHL.U32' else
                  f'R{4+i}, 0x4, RZ' if shift=='SHF.L.U32' else f'R20, 0x4, R{4+i}')
            seq.append(f'{shift} R{4+i}, {args}')
        seq+=['IMMA.16864.U4.S4 R4, R8.ROW, R12.COL, R4',
              'IMMA.16864.U4.S4 R4, R8.ROW, R12.COL, R4']
        for ins in seq:
            lines.append(f'/*{pc:04x}*/ {ins} ;'); pc+=16
    return '\n'.join(lines),{'loops':[dict(kind='integer',begin_pc='0x0',end_pc=hex(pc-16))]}


@pytest.mark.parametrize('op',['IMAD.SHL.U32','SHF.L.U32','SHF.L.W.U32.HI'])
def test_attribute_only_high_components_and_verify_low_consumption(op):
    sass,live=fixture_sass(op)
    r=high_reconstruction(sass,'candidate',live)
    assert r['components']==64 and r['opcodes']=={op:64} and r['low_mma_consumed_all']
    with pytest.raises(ValueError):
        high_reconstruction(sass.replace('0x4','0x3').replace('0x10','0x8'),'candidate',live)


def test_predeclared_gate_does_not_accept_unchanged_routing_or_work_spill():
    def live(n=383,imad=158,regs=168,local=0):
        return dict(allocated_gpr=regs,loops=[dict(kind='integer',static_instructions=n,
            opcode_counts={'IMAD':imad,'LDL':local,'IMMA.16864.S4.S4':32,
                           'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16,'LDGSTS.BYPASS.128':10})])
    route={'opcodes':{'SHF.L.W.U32.HI':64}}
    assert cost_gate(live(),live(imad=136),route)['passed']
    for candidate in (live(),live(imad=136,regs=169),live(imad=136,local=1),live(n=390,imad=136)):
        assert not cost_gate(live(),candidate,route)['passed']
    assert not cost_gate(live(),live(imad=136),{'opcodes':{'IMAD.SHL.U32':22,'SHF.L.U32':42}})['passed']
    assert LIMITS['min_fma_family_reduction']==.10 and LIMITS['max_static_work_ratio']==1.01


def test_full_sample_only_before_gpu_and_variant_rejection():
    from types import SimpleNamespace
    from benchmark_o78_high_funnel import require_full_samples,require_variant,timing_contract
    assert require_full_samples([])==['--samples','24']
    assert require_full_samples(['--samples=24'])==['--samples=24']
    with pytest.raises(SystemExit):
        require_full_samples(['--samples','4'])
    for v in ('o7','o8'):
        require_variant(SimpleNamespace(variant=v))
    for v in ('o3','o9'):
        with pytest.raises(ValueError):
            require_variant(SimpleNamespace(variant=v))
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        c=timing_contract(mode,100)
        assert not c['new_preparation_or_layout'] and c['supported_variants']==['o7','o8']
        assert c['stage_timing_inner_repeats']['total']==(100 if mode=='conversion_only' else 1)


def test_frozen_real_v78_high_dataflow_counts_not_address_shifts():
    import json
    p=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen'
    r=json.loads((p/'codegen.json').read_text())
    x=high_reconstruction((p/'o78_eight_chain.sass').read_text(),CONTROL,r['liveness'][CONTROL])
    assert x['components']==64 and x['opcodes']=={'SHF.L.U32':43,'IMAD.SHL.U32':21}

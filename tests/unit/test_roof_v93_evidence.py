"""Immutable v93 compile evidence; no GPU timing/MSE claim for rejected kernels."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from probe_stage_cycle_codegen import CONFIG, generated_header, cycle_liveness, worth_runtime

EVIDENCE = ROOT/'docs/evidence/a100_o378_roof_v93/reports'
PRIOR = ROOT/'docs/evidence/a100_o378_roof_v92/reports'
EXPECTED = {'o3':(990,330,79,77,184,496,336), 'o78':(770,385,31,30,120,240,164)}


@pytest.mark.parametrize('kind',('o3','o78'))
def test_audited_sources_and_all_retained_text_have_exact_digests(kind):
    cfg = CONFIG[kind]; directory = EVIDENCE/f'o378_roof_v93_{kind}_codegen'
    r = json.loads((directory/'codegen.json').read_text())
    digest = lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert r['source_commit'] == 'dce0b838a29811fbe99e80e6ce9ea5981e6a710a'
    assert r['scope'] == 'v93_fixed_stage_cycle_compile_gate'
    for path,sha in r['sources'].items():
        assert digest(ROOT/path) == sha
    for path,sha in r['artifact_sha256'].items():
        if Path(path).suffix == '.cubin':
            continue # complete binary archive is hash-identified in README, not committed
        assert digest(directory/path) == sha
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text() == generated_header(kind)
    assert not r['production_default_changed'] and not r['changed_semantics']


@pytest.mark.parametrize('kind',('o3','o78'))
def test_recompute_gate_and_normalized_work_from_actual_machine_code(kind):
    cfg = CONFIG[kind]; directory = EVIDENCE/f'o378_roof_v93_{kind}_codegen'
    r = json.loads((directory/'codegen.json').read_text())
    live = cycle_liveness((directory/'liveness.txt').read_text(),cfg['symbol'],cfg['stages'])
    assert live == r['liveness'][cfg['symbol']]
    loop = live['loops'][0]; counts = loop['opcode_counts']
    total,normalized,loads,stores,_,_,_ = EXPECTED[kind]
    assert loop['static_instructions'] == total
    assert loop['normalized_instructions_per_group'] == normalized
    assert sum(v for op,v in counts.items() if op.split('.')[0] == 'LDL') == loads
    assert sum(v for op,v in counts.items() if op.split('.')[0] == 'STL') == stores
    assert counts['IMMA.16864.S4.S4'] == counts['IMMA.16864.U4.S4'] == 32*cfg['stages']
    assert counts['LDSM.16.M88.4'] == 16*cfg['stages']
    assert live['allocated_gpr'] == 168
    assert not worth_runtime(r['liveness'][cfg['control']],live,cfg['stages'])
    assert r['worth_runtime_validation'] is False


@pytest.mark.parametrize('kind',('o3','o78'))
def test_encoded_old_control_preserved_and_new_entry_native_INT4(kind):
    cfg = CONFIG[kind]; directory = EVIDENCE/f'o378_roof_v93_{kind}_codegen'
    r = json.loads((directory/'codegen.json').read_text())
    prior = PRIOR/f'o378_roof_v92_{kind}_codegen'/f'{kind}_sixteen_chain.sass'
    sass = (directory/(cfg['stem']+'.sass')).read_text()
    assert compare(prior.read_text(),sass,'^'+cfg['control']+'$')['passed']
    assert r['control_comparison']['passed']
    entry = r['entries'][cfg['symbol']]
    assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
    assert entry['all_copies_bypass_l1']
    ptx = (directory/(cfg['stem']+'.ptx')).read_text()
    body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                if b.startswith('.visible .entry '+cfg['symbol']+'('))
    assert all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))


@pytest.mark.parametrize('kind',('o3','o78'))
def test_stack_and_ptxas_spill_not_confused_with_dynamic_bytes(kind):
    cfg = CONFIG[kind]; directory = EVIDENCE/f'o378_roof_v93_{kind}_codegen'
    _,_,_,_,stack,stores,loads = EXPECTED[kind]
    resource = (directory/'resources.txt').read_text()
    section = resource.split('Function '+cfg['symbol']+':',1)[1].split('Function',1)[0]
    assert f'REG:168 STACK:{stack}' in section
    log = (directory/'build.log').read_text().split('Function properties for '+cfg['symbol'],1)[1].split('ptxas info',1)[0]
    assert f'{stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads' in log
    assert not (EVIDENCE.parent/'runs').exists()

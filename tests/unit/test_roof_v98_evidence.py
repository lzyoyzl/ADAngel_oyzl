from pathlib import Path
import hashlib
import json
import re
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze as old_analyze
from probe_eight_warp_fullk_codegen import CONFIG, analyze, generated_header, generated_fallback, gate
from probe_roof_fullk_integer_codegen import static_entries

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v98'


@pytest.mark.parametrize('kind',('o3','o78'))
def test_hash_bound_static_work_geometry_and_rejected_gate(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/'reports'/f'o378_roof_v98_{kind}_codegen'
    receipt=json.loads((directory/'codegen.json').read_text())
    for name,digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            assert digest==receipt['cubin_sha256'];continue
        if name=='validate_coordinates':
            assert re.fullmatch('[0-9a-f]{64}',digest);continue
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    assert receipt['source_commit']=='ba5a927dec02b1685813581dfeee326de9808cb1'
    assert receipt['host_cute_coordinates_passed'] and receipt['threads']==256
    assert not receipt['changed_semantics'] and not receipt['production_default_changed']
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text()==generated_header(kind)
    assert (directory/(cfg['stem']+'_fallback_generated.cuh')).read_text()==generated_fallback(kind)
    text=(directory/'liveness.txt').read_text()
    old=old_analyze(text,cfg['control']);new=analyze(text,cfg['symbol'])
    assert receipt['liveness']=={cfg['control']:old,cfg['symbol']:new}
    assert new['allocated_gpr']==(121 if kind=='o3' else 128)
    actual_gate=gate(old,new)
    assert actual_gate==receipt['compile_gate'] and not actual_gate['passed']
    assert not receipt['worth_runtime_validation']
    loop=next(x for x in new['loops'] if x['kind']=='integer')
    assert loop['static_instructions']==(194 if kind=='o3' else 213)
    assert loop['max_live_gpr']==(112 if kind=='o3' else 113)
    assert actual_gate['hot_local_instructions']==0
    assert 1.20<actual_gate['weighted_static_instruction_ratio']<1.21 if kind=='o3' else (
        1.11<actual_gate['weighted_static_instruction_ratio']<1.12)


@pytest.mark.parametrize('kind',('o3','o78'))
def test_exact_old_encoding_same_entry_native_math_and_no_runtime_claim(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/'reports'/f'o378_roof_v98_{kind}_codegen'
    r=json.loads((directory/'codegen.json').read_text())
    sass=(directory/(cfg['stem']+'.sass')).read_text()
    old_root='a100_o378_roof_v89' if kind=='o3' else 'a100_o378_roof_v78'
    old=ROOT/'docs/evidence'/old_root/cfg['baseline']/(cfg['old_stem']+'.sass')
    assert compare(old.read_text(),sass,'^'+re.escape(cfg['control'])+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    symbols={cfg['control'],cfg['symbol']}
    assert static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)==r['entries']
    body=next(b for b in re.split(r'(?=\.visible \.entry )',(directory/(cfg['stem']+'.ptx')).read_text())
        if b.startswith('.visible .entry '+cfg['symbol']+'('))
    assert all(x in body for x in ('.maxntid 256, 1, 1','.minnctapersm 2',
        'cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    assert 'unique_outputs=8192 accumulator_per_thread=32' in (directory/'coordinates.log').read_text()
    doc=(ROOT/'docs/o3_o7_o8_eight_warp_fullk_20261006.md').read_text()
    assert '没有候选kernel launch' in doc and '不是本轮新测' in doc
    assert '不是实测负加速' in doc and '没有做Driver occupancy查询' in doc
    assert (EVIDENCE/'tmp/o378_v98_extension.sha256').read_text().startswith(
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')

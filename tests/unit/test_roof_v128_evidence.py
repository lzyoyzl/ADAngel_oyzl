"""Replay v128 static schedule/cost gate; there was no candidate GPU run."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v128'
B=E/'reports/o378_roof_v128_codegen'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_texts_sources_and_scope():
    index=read(E/'index.json');r=read(B/'codegen.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])==15
    assert index['artifact_bytes']==sum(x['bytes'] for x in index['files'])==10128266
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert r['source_commit']==index['compile_source_commit']=='67eff20593662f4c98b1d4ab549921ba67123c9d'
    for f,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',r['source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for f,digest in r['artifact_sha256'].items():
        if not f.endswith('.cubin'):assert sha(B/f)==digest
    assert not any(index[x] for x in ('candidate_GPU_executed','new_performance_or_MSE_results',
        'new_NCU_result','production_default_changed','candidate_adopted'))
    assert not r['production_default_changed'] and not r['conversion_changed'] and not r['changed_semantics']
    assert index['formal_extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_real_control_native_math_schedule_and_failed_gate():
    from probe_o3_crossk_load_codegen import CONTROL,SYMBOL,STEM,generated_header,overlap_evidence,cost_gate
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');sass=(B/(STEM+'.sass')).read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
        and e['all_copies_bypass_l1'] for e in r['entries'].values())
    live={s:analyze((B/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    evidence=overlap_evidence(sass,SYMBOL,live[SYMBOL])
    assert json.loads(json.dumps(evidence))==r['overlap']
    assert len(evidence['next_loads'])==8
    assert all(x['mma_before']==64 for x in evidence['next_loads'])
    assert evidence['next_loads'][0]['pc']=='0x60e0' and evidence['next_loads'][-1]['pc']=='0x6150'
    assert len(evidence['weighted_updates_after_first_next_load'])==32
    assert evidence['weighted_updates_after_first_next_load'][0]['pc']=='0x6170'
    gate=cost_gate(live[CONTROL],live[SYMBOL],evidence);gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and not gate['passed'] and not gate['checks']['work']
    assert all(v for k,v in gate['checks'].items() if k!='work')
    assert (gate['old_static'],gate['new_static'])==(323,374)
    a,b=(next(x for x in live[s]['loops'] if x['kind']=='integer') for s in (CONTROL,SYMBOL))
    assert live[CONTROL]['allocated_gpr']==live[SYMBOL]['allocated_gpr']==168
    assert (a['max_live_gpr'],b['max_live_gpr'])==(166,153)
    assert (a['opcode_counts']['LOP3.LUT'],b['opcode_counts']['LOP3.LUT'])==(4,21)
    assert (a['opcode_counts']['S2R'],b['opcode_counts']['S2R'])==(2,8)
    assert not any(op.startswith(('LDL','STL')) for op in b['opcode_counts'])
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header()
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',(B/(STEM+'.ptx')).read_text())
        if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    assert 'FAILED: no candidate GPU execution or adjacent schedule scan' in (E/'tmp/o378_v128_codegen.log').read_text()

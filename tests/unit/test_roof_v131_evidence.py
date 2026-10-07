"""Freeze an ineffective source schedule without inventing runtime results."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v131'
B=E/'reports/o378_roof_v131_codegen'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_sources_hashes_and_no_gpu_claims():
    index=read(E/'index.json');r=read(B/'codegen.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])==15
    assert index['artifact_bytes']==sum(f['bytes'] for f in index['files'])
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert r['source_commit']==index['compile_source_commit']=='212cde65ff6974e62fef956a29e04ba70dfccb48'
    for f,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',r['source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for f,digest in r['artifact_sha256'].items():
        if not f.endswith('.cubin'):assert sha(B/f)==digest
    assert not any(index[k] for k in ('candidate_GPU_executed','new_performance_or_MSE_results',
        'new_NCU_result','production_default_changed','candidate_adopted'))
    assert not any(r[k] for k in ('new_candidate_GPU_executed','production_default_changed','conversion_changed','changed_semantics'))


def test_same_exact_encoded_entry_and_failed_load_order_gate():
    from probe_o3_late_low_codegen import CONTROL,SYMBOL,STEM,generated_header,operand_load_order,cost_gate
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from compare_a100_codegen import compare,instructions
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');index=read(E/'index.json');sass=(B/(STEM+'.sass')).read_text()
    baseline=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    assert compare(baseline.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    pattern='^(?:'+CONTROL+'|'+SYMBOL+')$'
    words=instructions(sass,pattern)
    assert words[CONTROL]==words[SYMBOL] and len(words[CONTROL])//2==2200
    assert hashlib.sha256('\n'.join(words[SYMBOL]).encode()).hexdigest()==index['sass_words_sha256']
    es=static_entries(sass,pattern,{CONTROL,SYMBOL})
    assert es==r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values())
    live={s:analyze((B/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    orders={s:operand_load_order(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    chains={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    assert orders==r['load_order']
    # JSON serializes integer histogram keys as strings.
    assert json.loads(json.dumps(chains))==r['chains']
    gate=cost_gate(live[CONTROL],live[SYMBOL],orders[SYMBOL],chains[SYMBOL])
    gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and not gate['passed']
    assert {k for k,v in gate['checks'].items() if not v}=={'low_loads_really_delayed'}
    assert gate['old_static']==gate['new_static']==323
    for s in (CONTROL,SYMBOL):
        assert [x['mma_before'] for x in orders[s]['U4']]==[0,0,12,13]
        assert live[s]['allocated_gpr']==168 and chains[s]['peak_started_not_finished_chains']==8
        c=next(x for x in live[s]['loops'] if x['kind']=='integer')['opcode_counts']
        assert not any(op.startswith(('LDL','STL')) for op in c)
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header()
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',(B/(STEM+'.ptx')).read_text())
        if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))

"""Replay the new address repair, original failed gate and prior CPU error."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v134'
B=E/'reports/o378_roof_v134_layout_fixed_codegen'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())


def test_raw_artifacts_sources_and_non_GPU_scope():
    index=read(E/'index.json');r=read(B/'codegen.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])
    assert index['artifact_bytes']==sum(f['bytes'] for f in index['files'])
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert r['source_commit']==index['compile_source_commit']=='9795c7d2bd68d8d6b495df1464e0d3028df5e34d'
    for name,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',r['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for name,digest in r['artifact_sha256'].items():
        if name.endswith(('.json','.log','.cu','.cuh','.ptx','.sass','.txt')):assert sha(B/name)==digest
    assert not any(index[k] for k in ('candidate_GPU_executed','new_performance_or_MSE_results',
        'new_NCU_result','production_default_changed','candidate_adopted'))
    assert 'static assertion failed' in (E/'reports/o378_roof_v134_codegen/host_build.log').read_text()
    assert not (E/'reports/o378_roof_v134_codegen/codegen.json').exists()
    assert r['host_layout']==read(B/'host_layout.json')=={
        'passed':True,'source_addresses':3072,'destination_words':12288}


def test_real_native_math_overlap_and_original_failed_gate():
    from probe_o3_cached_a_iterator_codegen import CONTROL,SYMBOL,STEM,generated_header,overlap_evidence,cost_gate
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');sass=(B/(STEM+'.sass')).read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert es==r['entries'] and all(v['native_u4_s4'] and v['native_s4_s4'] and
        not v['int8_mma'] and v['all_copies_bypass_l1'] for v in es.values())
    live={s:analyze((B/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    ev=overlap_evidence(sass,SYMBOL,live[SYMBOL])
    assert json.loads(json.dumps(ev))==r['overlap']
    gate=cost_gate(live[CONTROL],live[SYMBOL],ev)
    gate['control_encoding_unchanged']=True
    hot=next(x for x in live[SYMBOL]['loops'] if x['kind']=='integer')
    gate['checks']['no_hot_shuffles']=not any(op.startswith('SHFL') for op in hot['opcode_counts'])
    gate['passed'] &= gate['checks']['no_hot_shuffles']
    assert gate==r['cost_gate'] and not gate['passed']
    assert {k for k,v in gate['checks'].items() if not v}=={'work','hot_local'}
    assert (gate['old_static'],gate['new_static'])==(323,354)
    assert live[SYMBOL]['allocated_gpr']==168 and hot['max_live_gpr']==158
    assert hot['opcode_counts']['LDL']==4 and not any(op.startswith('STL') for op in hot['opcode_counts'])
    assert (hot['opcode_counts']['S2R'],hot['opcode_counts']['LOP3.LUT'])==(2,10)
    assert len(ev['next_loads'])==8 and len(ev['weighted_updates_after_first_next_load'])==32
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header()
    assert 'FAILED: stop one address repair' in (E/'tmp/o378_v134_layout_fixed_codegen.log').read_text()

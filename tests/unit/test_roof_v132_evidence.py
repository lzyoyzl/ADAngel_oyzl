"""Replay a failed cached-coefficient gate, without manufacturing GPU results."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v132'
B=E/'reports/o378_roof_v132_codegen'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_sources_hashes_and_no_gpu_claims():
    index=read(E/'index.json');r=read(B/'codegen.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])==11
    assert index['artifact_bytes']==sum(f['bytes'] for f in index['files'])
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert r['source_commit']==index['compile_source_commit']=='81f6412e8ec2dc15e99917deaa5f50377400a547'
    for f,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',r['source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for f,digest in r['artifact_sha256'].items():
        if not f.endswith('.cubin'):assert sha(B/f)==digest
    assert not any(index[k] for k in ('candidate_GPU_executed','new_performance_or_MSE_results',
        'new_NCU_result','production_default_changed','candidate_adopted'))
    assert not any(r[k] for k in ('new_candidate_GPU_executed','production_default_changed','changed_semantics'))
    assert r['weight_conversion_must_include_table_build'] and r['table_range_guard_in_gemm']


def test_replay_actual_integer_loop_and_original_failed_gate():
    from probe_o7_cached_coeff_codegen import CONTROL,SYMBOL,STEM,generated_header,analyze_candidate,cost_gate
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');sass=(B/(STEM+'.sass')).read_text()
    baseline=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(baseline.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert es==r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values())
    live_text=(B/'liveness.txt').read_text()
    live={CONTROL:analyze(live_text,CONTROL),SYMBOL:analyze_candidate(live_text)}
    assert live==r['liveness']
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and not gate['passed']
    assert {k for k,v in gate['checks'].items() if not v}=={'work','integer_service'}
    assert (gate['old_static'],gate['new_static'],gate['old_imad'],gate['new_imad'])==(383,564,158,224)
    hot=next(x for x in live[SYMBOL]['loops'] if x['kind']=='cached_coeff_integer')['opcode_counts']
    assert hot['LDS']==71 and live[SYMBOL]['allocated_gpr']==168
    assert not any(op.startswith(('LDL','STL','STS','I2F','FFMA')) for op in hot)
    assert r['shared_bytes']==43520 and r['extra_weight_bytes_4096']==5242880
    assert (r['g128_input_bytes_old'],r['g128_input_bytes_new'])==(17152,21760)
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header()
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',(B/(STEM+'.ptx')).read_text())
        if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))

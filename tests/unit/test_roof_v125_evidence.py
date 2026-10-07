"""Replay v125 frozen audit and full24 negative pairing without GPU."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v125'
B=E/'reports/o378_roof_v125_codegen'
RUN=E/'runs/o378_roof_v125_full24'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_frozen_artifacts_source_and_binary_identities():
    index=read(E/'index.json');r=read(B/'codegen.json');env=read(RUN/'environment.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])==55
    for entry in index['files']:
        p=E/entry['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
    assert index['compile_source_commit']==r['source_commit']
    assert index['runtime_source_commit']==env['git_commit']
    for f,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',index['compile_source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    runtime=read(RUN/'build/split_weighted_build.json')
    for f,digest in runtime['runtime_sources'].items():
        raw=subprocess.check_output(['git','show',index['runtime_source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for name,digest in r['artifact_sha256'].items():
        if not name.endswith('.cubin'):assert sha(B/name)==digest
    assert env['extension_sha256']==index['formal_extension_sha256']==EXTENSION
    assert index['candidate_GPU_executed'] and index['new_performance_or_MSE_results']
    assert not any(index[x] for x in ('production_default_changed','candidate_adopted',
                                    'new_NCU_result','new_conversion_or_end_to_end_result'))


def test_replay_actual_native_chains_control_and_original_failed_gate():
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_split_weighted_codegen import CONTROL,SYMBOL,STEM,cost_gate,generated_header,split_chain_trace
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');sass=(B/(STEM+'.sass')).read_text()
    live={s:analyze((B/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    old=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==r['entries']
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header()
    chain=split_chain_trace(sass,SYMBOL,live[SYMBOL]);assert chain==r['chains']
    assert chain['chain_count']==32 and chain['mma_per_chain']==2 and chain['peak_started_not_finished']==8
    gate=cost_gate(live[CONTROL],live[SYMBOL],chain);gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and not gate['passed'] and not gate['checks']['hot_local']
    assert all(v for k,v in gate['checks'].items() if k!='hot_local')
    assert (gate['old_static'],gate['new_static'])==(323,337)
    a,b=(next(x for x in live[s]['loops'] if x['kind']=='integer') for s in (CONTROL,SYMBOL))
    assert a['opcode_counts']['IMAD']==65 and b['opcode_counts']['IMAD']==129
    assert b['opcode_counts']['LDL']==4 and not any(x.startswith('STL') for x in b['opcode_counts'])
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] for e in r['entries'].values())
    ptx=(B/(STEM+'.ptx')).read_text()
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',ptx) if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32','mad.lo.s32'))


def test_all24_three_rounds_unfiltered_events_and_MSE():
    from benchmark_a100_o1 import stats
    from benchmark_conversion_pipeline import summarize
    data=rows(RUN/'results.jsonl')
    ids={f'layer_{l:02d}_{p}_proj' for l in (0,6,12,18,24,31) for p in ('q','k','v','o')}
    expected={(sid,r,i) for sid in ids for r in range(3) for i in (0,1)}
    index={(r['sample_id'],r['round'],r['implementation']):r for r in data}
    assert len(data)==len(index)==144 and set(index)==expected
    for r in data:
        assert r['mode']=='compute_only' and r['variant']=='o3' and r['paired_reference']=='o0'
        assert r['bitwise_equal_current_best'] and r['payload_bitwise'] and r['mse_vs_current_best']==0
        assert r['weight_cached'] and r['activation_prepared']
        assert r['total_timing']=='single_execution_cuda_event'
        assert r['stage_timing_inner_repeats']=={'gemm':1,'total':1}
        assert set(r['raw_ms'])=={'gemm','total'}
        for name,values in r['raw_ms'].items():
            assert len(values)==200 and min(values)>0
            assert stats(values)==pytest.approx(r['stage_summaries'][name])
        mate=index[(r['sample_id'],r['round'],1-r['implementation'])]
        assert r['guard_status']==mate['guard_status'] and r['mse_vs_paired_fp16']==mate['mse_vs_paired_fp16']
    s=read(RUN/'summary.json')
    assert s['records']==summarize(data) and not s['production_default_changed']
    old,new=s['records']
    assert old['samples']==new['samples']==24 and old['records']==new['records']==72
    assert old['median_ms']==pytest.approx(.43673600256443024)
    assert new['median_ms']==pytest.approx(.46592000126838684)
    assert new['paired_speedup']==pytest.approx(.9428571556500354)
    assert new['paired_speedup_ci95'][1]<1
    assert old['selected_cv_failed_records']==new['selected_cv_failed_records']==1
    assert old['median_mse_vs_paired_fp16']==new['median_mse_vs_paired_fp16']==.006653010287409885
    assert old['mean_mse_vs_paired_fp16']==new['mean_mse_vs_paired_fp16']==.007578847013302748


def test_resources_synthetic_and_limited_sanitizer_scope():
    env=read(RUN/'environment.json');resources=read(RUN/'resources.json')
    assert env['no_small_performance_screen'] and env['small_spill_review_before_GPU_execution']
    assert not env['original_compile_gate_passed'] and not env['production_default_changed']
    for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100,modes=['compute_only']).items():
        assert env['args'][k]==v
    for name,size in (('v89',16),('v125',24)):
        r=resources[name]
        assert r['registers_per_thread']==168 and r['active_blocks_per_sm']==3
        assert r['shared_memory_bytes']==50688 and r['threads']==128 and r['local_size_bytes']==size
    build=read(RUN/'build/split_weighted_build.json')
    assert build['review_precedes_candidate_GPU_execution'] and build['preparation_identical']
    assert build['initial_zero_spill_gate']==read(B/'codegen.json')['cost_gate']
    for name in ('full24','memcheck','synccheck'):
        v=read(E/f'runs/o378_roof_v125_{name}/validation.json')
        assert v['passed'] and len(v['checks'])==96 and len(v['rejected'])==8
        assert len(v['grouped_coordinates_checks'])==2
        assert all(c['finite_fp32'] and c['payload_exact'] and c['scale_exact']
                   and c['semantic_tolerance_passed'] and c['bitwise_v61'] for c in v['checks'])
    for tool in ('memcheck','synccheck'):
        log=(E/f'tmp/o378_v125_{tool}.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in log and 'VALIDATION PASSED' in log
        assert 'No kernels were profiled' not in log

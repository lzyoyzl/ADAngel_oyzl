"""Replay frozen v127 compilation, numerical checks and full24 pairing, no GPU."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v127'
B=E/'reports/o378_roof_v127_codegen'
RUN=E/'runs/o378_roof_v127_full24'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_raw_hashes_provenance_and_unchanged_production():
    index=read(E/'index.json');r=read(B/'codegen.json');env=read(RUN/'environment.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])
    for item in index['files']:
        f=E/item['path']
        assert f.relative_to(ROOT).as_posix() in tracked
        assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256']
    assert index['compile_source_commit']==r['source_commit']
    assert index['runtime_source_commit']==env['git_commit']
    for name,digest in r['sources'].items():
        content=subprocess.check_output(['git','show',r['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==digest
    for name,digest in r['artifact_sha256'].items():
        if not name.endswith('.cubin'):assert sha(B/name)==digest
    review=read(E/'runs/o378_roof_v127_full24_preexecution_review.json')
    for name,digest in review['runtime_sources'].items():
        content=subprocess.check_output(['git','show',index['runtime_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(content).hexdigest()==digest
    assert review['review_precedes_this_process_candidate_GPU_execution']
    assert review['codegen']==r
    assert env['extension_sha256']==index['formal_extension_sha256']==EXTENSION
    assert index['candidate_GPU_executed'] and index['new_performance_or_MSE_results']
    assert not env['production_default_changed'] and not index['production_default_changed']


def test_native_math_schedule_and_preserved_failed_gate():
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from compare_a100_codegen import compare
    from probe_o78_activation_panel_codegen import CONTROL,SYMBOL,STEM,generated_header,cost_gate
    from probe_roof_fullk_integer_codegen import static_entries
    from benchmark_o78_activation_panel import same_json_value,resource_review
    r=read(B/'codegen.json');sass=(B/(STEM+'.sass')).read_text()
    live={s:analyze((B/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==r['entries']
    assert (B/(STEM+'_generated.cuh')).read_text()==generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    schedules={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    assert same_json_value(schedules,r['schedules'])
    assert [schedules[s]['peak_started_not_finished_chains'] for s in (CONTROL,SYMBOL)]==[8,6]
    gate=cost_gate(live[CONTROL],live[SYMBOL],r['runtime_resources'])
    gate['checks']['control_encoding_unchanged']=True;gate['passed']=all(gate['checks'].values())
    assert gate==r['cost_gate'] and not gate['passed']
    assert (gate['old_static'],gate['new_static'])==(383,356)
    review=resource_review(gate,live[SYMBOL],r['runtime_resources'])
    assert review['passed'] and not review['original_zero_spill_gate']['passed']
    a,b=(next(x for x in live[s]['loops'] if x['kind']=='integer') for s in (CONTROL,SYMBOL))
    assert a['opcode_counts']['IMAD']==128 and b['opcode_counts']['IMAD']==131
    assert b['opcode_counts']['LDL']==1 and not any(k.startswith('STL') for k in b['opcode_counts'])
    ptx=(B/(STEM+'.ptx')).read_text()
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',ptx) if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] for e in r['entries'].values())


def test_full24_all_events_MSE_and_safety():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    data=rows(RUN/'results.jsonl');env=read(RUN/'environment.json')
    ids={f'layer_{l:02d}_{p}_proj' for l in (0,6,12,18,24,31) for p in ('q','k','v','o')}
    index={(r['sample_id'],r['variant'],r['round'],r['candidate']):r for r in data}
    assert len(data)==len(index)==288
    assert set(index)=={(s,v,r,p) for s in ids for v in ('o7','o8') for r in range(3) for p in (0,1)}
    for key,row in index.items():
        assert row['bitwise_equal_v67'] and row['MSE_regression_passed'] and row['metadata_exact']
        assert row['finite_fp32'] and row['mse_vs_v67']==row['max_abs_vs_v67']==0
        assert row['mode']=='compute_only' and row['conversion_unchanged']
        assert row['activation_panel_startup_in_GEMM'] and not row['original_compile_gate_passed']
        assert row['total_timing']=='single_execution_cuda_event'
        assert row['stage_timing_inner_repeats']=={'gemm':1,'total':1}
        assert row['weight_cached'] and row['activation_prepared']
        assert row['raw_ms']['gemm']==row['raw_ms']['total']
        for stage,values in row['raw_ms'].items():
            assert len(values)==200 and min(values)>0
            assert stats(values)==pytest.approx(row['stage_summaries'][stage])
        mate=index[(*key[:3],1-key[3])]
        assert row['guard']==mate['guard'] and row['mse_vs_paired_fp16']==mate['mse_vs_paired_fp16']
    summary=read(RUN/'summary.json')['records']
    assert summary==summarize(data,('compute_only',))
    for variant,latencies,speed,mse in (
        ('o7',(.47411200404167175,.48230400681495667),.982923734307215,.005536172273439442),
        ('o8',(.47513601183891296,.4833280146121979),.9828141524309477,.004411084910985704)):
        pair=[r for r in summary if r['variant']==variant]
        assert [r['median_ms'] for r in pair]==pytest.approx(latencies)
        assert pair[1]['paired_speedup']==pytest.approx(speed) and pair[1]['paired_speedup_ci95'][1]<1
        assert pair[0]['median_mse']==pair[1]['median_mse']==mse
    assert env['no_filtering'] and not env['production_default_changed']
    for k,v in dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100).items():assert env['args'][k]==v
    for p,shared,local in (('0',34304,0),('1',41984,8)):
        r=env['resources'][p]
        assert r['shared_memory_bytes']==shared and r['local_size_bytes']==local
        assert r['registers_per_thread']==168 and r['active_blocks_per_sm']==3
    for name in ('validate','memcheck','synccheck','full24'):
        v=read(E/f'runs/o378_roof_v127_{name}/validation.json')
        assert v['passed'] and v['count']==len(v['checks'])==64 and v['edge_count']==12
        assert len(v['activation_panel_mapping_checks'])==2
        assert all(c['bitwise_v78'] and c['finite_fp32'] and c['semantic_tolerance_passed'] and
            c['activation_factors_vary_across_rows_and_groups'] and c['integer_ctas']>0
            for c in v['activation_panel_mapping_checks'])
    for tool in ('memcheck','synccheck'):
        log=(E/f'tmp/o378_v127_{tool}.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in log and 'VALIDATION PASSED' in log
        assert 'No kernels were profiled' not in log

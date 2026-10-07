"""Replay frozen fixed-high routing and full24 pairing without GPU or compile."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v117'
B=E/'reports/o378_roof_v117_codegen'
RUN=E/'reports/o378_roof_v117_compute24'
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def test_complete_raw_artifacts_and_original_sources():
    i=read(E/'index.json')
    assert i['artifact_count']==len(i['artifacts'])
    for name,digest in i['artifacts'].items():
        assert sha(E/name)==digest,name
    for name in i['binary_files_not_committed']:
        assert not (E/name).exists()
    assert i['candidate_GPU_launched'] and i['new_performance_result']
    assert not i['production_default_changed'] and not i['candidate_adopted']
    r=read(B/'codegen.json');env=read(RUN/'environment.json')
    assert r['source_commit']==i['compile_source_commit']
    assert env['git_commit']==i['runtime_source_commit']
    for name,digest in r['sources'].items():
        original=subprocess.check_output(['git','show',i['compile_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    for name,digest in env['codegen']['high_funnel']['runtime_source_sha256'].items():
        original=subprocess.check_output(['git','show',i['runtime_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    assert env['extension_sha256']==i['formal_extension_sha256_after']==EXTENSION


def test_actual_control_code_high_dataflow_and_cost_gate():
    from compare_a100_codegen import compare
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from probe_o78_high_funnel_codegen import CONTROL,SYMBOL,cost_gate,high_reconstruction,generated_header
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');sass=(B/'o78_high_funnel.sass').read_text()
    old=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old,sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==r['entries']
    for path,digest in r['artifact_sha256'].items():assert sha(B/path)==digest,path
    header=B/'o78_high_funnel_generated.cuh'
    assert sha(header)==r['generated_header_sha256']
    assert header.read_text()==generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    for s,n,peak in ((CONTROL,383,8),(SYMBOL,385,6)):
        live=analyze((B/'liveness.txt').read_text(),s)
        assert live==r['liveness'][s] and live['allocated_gpr']==168
        loop=next(x for x in live['loops'] if x['kind']=='integer')
        assert loop['static_instructions']==n
        assert not any(op.startswith(('LDL','STL')) for op in loop['opcode_counts'])
        chain=trace(sass,s,live)
        assert chain['total_mma']==64 and chain['peak_started_not_finished_chains']==peak
        assert high_reconstruction(sass,s,live)==r['high_reconstruction'][s]
        body=next(t for t in re.split(r'(?=\.visible \.entry )',(B/'o78_high_funnel.ptx').read_text())
            if t.startswith('.visible .entry '+s+'('))
        assert all(t in body for t in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    assert r['high_reconstruction'][CONTROL]['opcodes']=={'SHF.L.U32':43,'IMAD.SHL.U32':21}
    assert r['high_reconstruction'][SYMBOL]['opcodes']=={'SHF.L.W.U32.HI':64}
    gate=cost_gate(r['liveness'][CONTROL],r['liveness'][SYMBOL],r['high_reconstruction'][SYMBOL])
    gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and gate['passed']
    assert gate['imad_family_reduction']==pytest.approx(21/158)


def test_all24_three_round_pairs_raw_events_MSE_and_unfiltered_summary():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_high_funnel import timing_contract
    data=rows(RUN/'results.jsonl')
    ids={f'layer_{layer:02d}_{proj}_proj' for layer in (0,6,12,18,24,31) for proj in ('q','k','v','o')}
    assert len(data)==288 and {r['sample_id'] for r in data}==ids
    assert len({(r['sample_id'],r['variant'],r['round'],r['candidate']) for r in data})==288
    assert {r['round'] for r in data}=={0,1,2} and {r['candidate'] for r in data}=={0,1}
    for r in data:
        assert r['mode']=='compute_only' and r['paired_fp16']=={'o7':'o5','o8':'o6'}[r['variant']]
        assert all(r[k] for k in ('finite_fp32','metadata_exact','bitwise_equal_v67','MSE_regression_passed'))
        assert r['mse_vs_v67']==r['max_abs_vs_v67']==0
        assert r['guard']['ctas']==2048 and r['guard']['invalid_ctas']==0
        assert r['guard']['integer_ctas']+r['guard']['fallback_ctas']==2048
        fallback=12 if (r['variant'],r['sample_id'])==('o8','layer_24_o_proj') else 0
        assert r['guard']['fallback_ctas']==fallback
        assert set(r['raw_ms'])=={'gemm','total'} and r['raw_ms']['gemm']==r['raw_ms']['total']
        for key,value in timing_contract('compute_only',100).items():assert r[key]==value
        for stage,times in r['raw_ms'].items():
            assert len(times)==200 and min(times)>0
            assert stats(times)==pytest.approx(r['stage_summaries'][stage])
    summary=read(RUN/'summary.json')
    paired={(r['sample_id'],r['variant'],r['round'],r['candidate']):r for r in data}
    for r in data:
        mate=paired[(r['sample_id'],r['variant'],r['round'],1-r['candidate'])]
        assert r['guard']==mate['guard']
    assert summary['records']==summarize(data,('compute_only',),('o7','o8'))
    assert summary['no_filtering'] and not summary['production_default_changed']
    for v in ('o7','o8'):
        old,new=[r for r in summary['records'] if r['variant']==v]
        assert old['samples']==new['samples']==24 and old['records']==new['records']==72
        assert old['median_mse']==new['median_mse'] and old['mean_mse']==new['mean_mse']
        assert new['paired_speedup_ci95'][1]<1  # observed negative route, not adopted


def test_scope_same_preparation_resources_and_synthetic_edges():
    env=read(RUN/'environment.json');v=read(RUN/'validation.json')
    assert v['passed'] and v['count']==64 and v['edge_count']==12
    assert v['no_small_performance_screen'] and v['quantization_and_scale_unchanged']
    assert env['variants']==['o7','o8'] and env['args']['samples']==24
    assert env['args']['rounds']==3 and env['args']['warmup']==1000
    assert env['args']['repeats']==200 and env['args']['inner']==100
    assert env['source_quantization']=='original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope']=='cached_compute_only_not_E2E' and env['no_filtering']
    assert not env['production_default_changed']
    preparation=read(ROOT/'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen/build.json')
    assert env['gpu_preparation_build']==preparation
    for p in ('0','1'):
        r=env['resources'][p]
        assert r['registers_per_thread']==168 and r['local_size_bytes']==0 and r['threads']==128
        assert r['active_blocks_per_sm']==3 and r['shared_memory_bytes']==34304
        assert r['cta_tile']==[64,128,128] and r['pipeline_stages']==2
    source=rows(RUN/'source_provenance.jsonl')
    assert len(source)==len({(r['sample_id'],r['variant']) for r in source})==48
    for r in source:
        assert re.fullmatch('[0-9a-f]{64}',r['raw_sha256'])
        for k in ('activation','weight'):
            assert r[k]['group_size']==128 and r[k]['shape']==[4096,4096]
    assert not read(E/'index.json')['new_conversion_or_end_to_end_result']

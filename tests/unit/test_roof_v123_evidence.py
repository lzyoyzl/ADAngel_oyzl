"""CPU replay of v123 frozen sources, SASS, raw Events, identity and MSE."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v123'
B=E/'reports/o378_roof_v123_codegen_r2'
RUN=E/'runs/o378_v123_full24'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(line) for line in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_raw_files_sources_and_unchanged_extension():
    index=read(E/'index.json');build=read(B/'build.json');env=read(RUN/'environment.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert len(index['files'])==index['artifact_count']
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert build['source_commit']==index['compile_source_commit']
    assert env['git_commit']==index['runtime_source_commit']
    assert env['extension_sha256']==index['formal_extension_sha256']
    assert index['formal_extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    for name,digest in build['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',index['compile_source_commit']+':'+name],cwd=ROOT)).hexdigest()==digest
    for name,digest in build['artifact_sha256'].items():
        if name!='libo78_gpu_prepare.so':assert sha(B/name)==digest
    assert not index['production_default_changed'] and not index['GEMM_modified']


def test_original_failed_gate_resource_review_native_DP4A_and_old_encoding():
    import probe_nv6_swar_codegen as probe
    from compare_a100_codegen import compare
    from analyze_o78_row_fused_codegen import entries
    original=read(E/'reports/o378_roof_v123_codegen/build.json')
    assert not original['audit']['worth_runtime_validation']
    build=read(B/'build.json');audit=build['audit']
    old=ROOT/'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen/prepare.sass'
    actual=entries((B/'prepare.sass').read_text())
    for label in ('control','candidate'):
        item=audit[label];parsed=actual[item['symbol']]
        assert item['opcode_counts']==parsed['opcode_counts'] and item['instructions']==parsed['instructions']
        assert item['stack']==item['local']==0 and item['shared']==256
        assert audit['runtime_resources'][label]['active_blocks_per_sm']==8
    assert (audit['control']['instructions'],audit['candidate']['instructions'])==(696,408)
    assert (audit['control']['registers'],audit['candidate']['registers'])==(23,29)
    assert audit['candidate']['opcode_counts']['IDP.4A.S8.S8']==4
    assert audit['static_instruction_reduction_fraction']==pytest.approx(288/696)
    assert not audit['first_no_register_growth_gate_passed']
    assert audit['resource_review_before_any_kernel_launch'] and audit['residency_equal']
    assert audit['worth_runtime_validation'] and audit['old_controls']['passed']
    import re
    old_symbols=entries(old.read_text())
    assert compare(old.read_text(),(B/'prepare.sass').read_text(),
        '|'.join('^'+re.escape(s)+'$' for s in old_symbols))==audit['old_controls']
    # Resource-query recompilation must not change the actual candidate code.
    first=entries((E/'reports/o378_roof_v123_codegen/prepare.sass').read_text())
    before=next(e for s,e in first.items() if 'adangel_sm80_nv6_swar_metadata' in s)
    assert before['encoded_words']==actual[audit['candidate']['symbol']]['encoded_words']
    assert (B/'nv6_swar_generated.cuh').read_text()==probe.generated_header()
    assert (B/'nv6_swar_prepare.cu').read_text()==probe.generated_host()


def test_full24_four_modes_unfiltered_stats_MSE_and_source():
    from analyze_nv6_swar import analyze
    from benchmark_o8_nv6_swar import timing_contract
    result=analyze(RUN)
    assert result==read(RUN/'analysis.json')
    assert result['records']==576 and result['original_event_values']==345600
    assert result['all_outputs_bitwise_equal'] and not result['GEMM_modified']
    assert result['output_mse_vs_O6']==pytest.approx(dict(median=.004411084910985704,mean=.004381379299073540))
    source=rows(RUN/'source_provenance.jsonl')
    old=rows(ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')
    assert source==[r for r in old if r['variant']=='o8']
    env=read(RUN/'environment.json')
    assert (env['args']['samples'],env['args']['rounds'],env['args']['warmup'],env['args']['repeats'],env['args']['inner'])==(24,3,1000,200,100)
    for r in rows(RUN/'results.jsonl'):
        assert r['paired_fp16']=='o6' and r['guard']['invalid_ctas']==0
        for key,value in timing_contract(r['mode'],100).items():assert r[key]==value
        assert r['resources']['kernel_symbol']=='adangel_roof_o78_eight_chain_candidate'
        assert r['resources']['registers_per_thread']==168 and r['resources']['active_blocks_per_sm']==3


def test_synthetic_guard_word_checks_and_limited_sanitizers():
    validation=read(RUN/'validation.json')
    assert validation['passed'] and validation['count']==32 and validation['edge_count']==3
    assert validation['packed_word_exhaustive']['words']==131072
    assert validation['packed_word_exhaustive']['passed']
    for name in ('preflight','memcheck','synccheck'):
        assert read(E/f'runs/o378_v123_{name}/validation.json')==validation
    for tool in ('memcheck','synccheck'):
        log=(E/f'tmp/o378_v123_{tool}.log').read_text()
        assert 'FP6 PACKED SWAR VALIDATION PASSED' in log and 'ERROR SUMMARY: 0 errors' in log

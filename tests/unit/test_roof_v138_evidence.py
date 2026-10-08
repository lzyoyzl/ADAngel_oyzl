"""Frozen v138 complete evidence: exact files, SASS, all Events, MSE and scope."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v138'
B=E/'reports/o378_roof_v138_codegen_r2'
RUN=E/'runs/o378_v138_completed24'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_hashes_sources_and_unchanged_extension():
    index=read(E/'index.json');build=read(B/'build.json');env=read(RUN/'environment.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert len(index['files'])==index['artifact_count']
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert not list(E.rglob('*.so')) and not list(E.rglob('*.cubin'))
    assert sum(f['bytes'] for f in index['files'])==index['artifact_bytes']
    assert sha(E/'analysis.json')==index['analysis_sha256']
    assert build['source_commit']==index['compile_source_commit']
    assert env['git_commit']==index['runtime_source_commit']
    assert env['extension_sha256']==index['formal_extension_sha256']
    for name,digest in build['sources'].items():
        src=subprocess.check_output(['git','show',index['compile_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(src).hexdigest()==digest
    runtime=subprocess.check_output(['git','show',index['runtime_source_commit']+':scripts/benchmark_o8_hif4_swar.py'],cwd=ROOT)
    assert hashlib.sha256(runtime).hexdigest()==env['codegen']['runtime_source_sha256']
    recovery=read(RUN/'recovery.json')
    for field,path in (('recovery_script_sha256','scripts/resume_hif4_swar.py'),
                       ('common_protocol_sha256','scripts/benchmark_o78_coefficient_probe.py')):
        source=subprocess.check_output(['git','show',recovery['resumed_git_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==recovery[field]
    for name,digest in build['artifact_sha256'].items():
        if name!='libo78_gpu_prepare.so':assert sha(B/name)==digest


def test_exact_controls_and_new_conversion_machine_code():
    from analyze_o78_row_fused_codegen import entries
    from compare_a100_codegen import compare
    import probe_hif4_swar_codegen as probe
    build=read(B/'build.json');audit=build['audit'];sass=(B/'prepare.sass').read_text()
    oldpath=ROOT/'docs/evidence/a100_o378_roof_v123/reports/o378_roof_v123_codegen_r2/prepare.sass'
    old=oldpath.read_text();pattern='|'.join('^'+re.escape(s)+'$' for s in entries(old))
    assert compare(old,sass,pattern)==audit['raw_symbol_comparison']
    for actual,expected in audit['anonymous_probe_symbol_mapping'].items():sass=sass.replace(actual,expected)
    assert compare(old,sass,pattern)==audit['old_controls']
    assert audit['old_controls']['passed'] and audit['old_controls']['old_symbols']==21
    parsed=entries((B/'prepare.sass').read_text())
    for key in ('control','candidate'):
        r=audit[key]
        for field in ('instructions','encoded_words','opcode_counts'):
            assert r[field]==parsed[r['symbol']][field]
        assert r['local']==r['stack']==0 and r['shared']==256
        assert audit['runtime_resources'][key]['active_blocks_per_sm']==8
    assert (audit['control']['instructions'],audit['candidate']['instructions'])==(544,312)
    assert (audit['control']['registers'],audit['candidate']['registers'])==(31,22)
    assert audit['scalar_DP4A_instructions']==4 and audit['cta_barriers']==1
    assert audit['worth_runtime_validation'] and audit['activation_encoding_identical']
    initial=read(E/'reports/o378_roof_v138_codegen/build.json')['audit']
    assert not initial['worth_runtime_validation'] and not initial['old_controls']['changed']
    for name,source in probe.generated_files().items():assert (B/name).read_text()==source


def test_full24_four_mode_replay_mse_and_safety():
    from freeze_hif4_swar_evidence import validate_frozen
    from benchmark_o8_hif4_swar import timing_contract
    result=validate_frozen(E)
    assert result==read(E/'analysis.json')
    assert result['records']==576 and result['original_event_values']==345600
    assert result['all_outputs_bitwise_equal'] and not result['all_real_integer_path']
    assert result['guard_summary']==dict(total_ctas_once_per_sample=49152,
        integer_ctas_once_per_sample=49140,fallback_ctas_once_per_sample=12,
        fallback_samples={'layer_24_o_proj':12},same_guard_all_policies_modes_rounds=True)
    assert result['output_mse_vs_O6']==pytest.approx(dict(median=.004411084910985704,mean=.004381379299073540))
    assert result['decoder_words']==1048576 and result['validation_cases']==32 and result['guard_edges']==3
    for row in map(json.loads,(RUN/'results.jsonl').read_text().splitlines()):
        for key,value in timing_contract(row['mode'],100).items():assert row[key]==value
        assert row['resources']['kernel_symbol']=='adangel_roof_o78_eight_chain_candidate'
        assert row['resources']['registers_per_thread']==168 and row['resources']['active_blocks_per_sm']==3
    # The same audited native-INT4 cubin is opened for both runtime policies.
    env=read(RUN/'environment.json')
    audit=env['codegen']['best_GEMM']['eight_chain']['build']
    prior=read(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/codegen.json')
    assert audit['cubin_sha256']==prior['cubin_sha256']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] for e in audit['entries'].values())

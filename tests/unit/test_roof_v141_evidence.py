"""Replay v141 compiler facts, all576 paired records and negative acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'python'))
from freeze_recycled_coefficient_evidence import verify,BUILD,EXTENSION
from probe_o78_recycled_coefficient_codegen import CONTROL,SYMBOL,STEM,generated_header

E=ROOT/'docs/evidence/a100_o378_roof_v141'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def test_all_artifacts_sources_and_binary_identity():
    index=read(E/'index.json');build=read(E/BUILD/'codegen.json')
    assert index['artifact_count']==len(index['artifacts'])
    for name,digest in index['artifacts'].items():assert sha(E/name)==digest,name
    for name,digest in index['binary_files_not_committed'].items():
        assert not (E/name).exists()
        assert digest==build['cubin_sha256']
    assert sha(E/'analysis.json')==index['analysis_sha256']
    assert index['candidate_GPU_launched'] and index['new_performance_result']
    assert not index['candidate_adopted'] and not index['production_default_changed']
    for name,digest in build['sources'].items():
        original=subprocess.check_output(['git','show',index['compile_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    env=read(E/'runs/o378_roof_v141_paired/environment.json')
    for name,digest in env['codegen']['recycled_coefficient']['runtime_source_sha256'].items():
        original=subprocess.check_output(['git','show',index['runtime_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    assert env['extension_sha256']==index['formal_extension_sha256_after']==EXTENSION


def test_control_native_math_and_bounded_header():
    from compare_a100_codegen import compare
    from probe_roof_fullk_integer_codegen import static_entries
    b=E/BUILD;r=read(b/'codegen.json');sass=(b/(STEM+'.sass')).read_text()
    old=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old,sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed'] and r['compile_gate']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==r['entries']
    assert (b/(STEM+'_generated.cuh')).read_text()==generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    for s,count,peak in ((CONTROL,383,166),(SYMBOL,372,164)):
        live=r['liveness'][s];loop=next(x for x in live['loops'] if x['kind']=='integer')
        assert live['allocated_gpr']==168 and loop['static_instructions']==count and loop['max_live_gpr']==peak
        assert not any(op.startswith(('LDL','STL')) for op in loop['opcode_counts'])
        assert r['entries'][s]['native_s4_s4'] and r['entries'][s]['native_u4_s4'] and not r['entries'][s]['int8_mma']


def test_full_raw_replay_mse_resources_and_stability_boundary():
    result=verify(E)
    assert result==read(E/'analysis.json')
    assert result['scale_overlap'][CONTROL]['old_slice_updates_after_next_mma']==16
    assert result['scale_overlap'][SYMBOL]['old_slice_updates_after_next_mma']==1
    assert result['scale_overlap'][CONTROL]['next_slice_mma_before_old_scale_drains']==7
    assert result['scale_overlap'][SYMBOL]['next_slice_mma_before_old_scale_drains']==1
    assert sum(r['records'] for r in result['runs'].values())==576
    for name,run in result['runs'].items():
        assert run['records']==288 and run['no_filtering'] and not run['conversion_modified']
        for row in run['table']:
            assert row['old_cv_failed']==row['new_cv_failed']==72
            assert not row['all_record_CV_below3'] and not row['median_speedup_ci_above_one']
            assert row['paired_speedup_ci95'][1]<1 and row['output_bitwise_unchanged']
            assert row['median_mse']==pytest.approx({'o7':0.005536172273439442,'o8':0.004411084910985704}[row['variant']])
        env=read(E/f'runs/{name}/environment.json')
        for policy in ('0','1'):
            r=env['resources'][policy]
            assert (r['registers_per_thread'],r['local_size_bytes'],r['active_blocks_per_sm'])==(168,0,3)
            assert r['threads']==128 and r['shared_memory_bytes']==34304
    assert result['synthetic_checks']==64 and result['edge_checks']==12
    assert result['limited_sanitizers']['memcheck']==result['limited_sanitizers']['synccheck']=='0 errors'


def test_same48_source_identities_and_actual_guard_coverage():
    old=rows(ROOT/'docs/evidence/a100_o378_roof_v117/reports/o378_roof_v117_compute24/source_provenance.jsonl')
    identity=lambda r:(r['sample_id'],r['variant'])
    old={identity(r):r for r in old}
    for name in ('o378_roof_v141_paired','o378_roof_v141_paired_r2'):
        source=rows(E/f'runs/{name}/source_provenance.jsonl')
        assert len(source)==48 and {identity(r):r for r in source}==old
        for row in rows(E/f'runs/{name}/results.jsonl'):
            count=12 if (row['variant'],row['sample_id'])==('o8','layer_24_o_proj') else 0
            assert row['guard']['fallback_ctas']==count
            assert row['mse_vs_v67']==row['max_abs_vs_v67']==0

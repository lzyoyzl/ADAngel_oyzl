"""Recompute complete conversion evidence; do not infer a GEMM speedup."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_mx8_warp_lut_codegen as probe
from analyze_mx8_warp_lut import analyze
from analyze_o78_row_fused_codegen import entries

BASE=ROOT/'docs/evidence/a100_o378_roof_v106'
BUILD=BASE/'reports/o378_roof_v106_codegen'
DATA=BASE/'runs/o378_roof_v106_full24'
EXTENSION_SHA='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def read(path):
    return json.loads(path.read_text())


def test_complete_raw_events_analysis_and_no_best_switch():
    result=analyze(DATA)
    assert result==read(DATA/'analysis.json')
    assert result['samples']==24 and result['records']==576
    assert result['original_event_values']==345600 and result['no_filtering']
    assert result['all_outputs_bitwise_equal']
    assert not result['GEMM_modified'] and not result['production_default_changed']
    stages={(r['mode'],r['stage']):r for r in result['stages']}
    activation=stages['conversion_only','activation_conversion']
    conversion=stages['conversion_only','total']
    assert activation['paired_speedup']==pytest.approx(1.0157533119187616,rel=1e-12)
    assert conversion['paired_speedup']==pytest.approx(1.0102514285753001,rel=1e-12)
    assert activation['paired_speedup_ci95'][0]>1 and conversion['paired_speedup_ci95'][0]>1
    assert conversion['control_cv_failed']==conversion['candidate_cv_failed']==0
    assert stages['compute_only','gemm']['control_cv_failed']==71
    assert stages['compute_only','gemm']['candidate_cv_failed']==72
    assert result['conclusion']['keep_existing_best']
    assert not result['conclusion']['GEMM_speedup_claim']
    assert not result['conclusion']['end_to_end_gain_confirmed']
    assert result['output_mse_vs_O5']==dict(median=.005536172273439442,mean=.005053635851002639)


def test_codegen_gate_and_old_encoded_entries_reproduce(monkeypatch):
    receipt=read(BUILD/'build.json')
    monkeypatch.setattr(probe,'BASELINE',ROOT/'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen')
    actual=probe.audit(BUILD)
    assert actual==receipt['audit'] and actual['worth_runtime_validation']
    assert actual['old_controls']['passed']
    assert actual['control']['instructions']==568 and actual['candidate']['instructions']==456
    assert actual['control']['registers']==32 and actual['candidate']['registers']==31
    assert actual['candidate']['shared']==256
    assert actual['candidate']['stack']==actual['candidate']['local']==actual['local_instructions']==0
    assert actual['cta_barriers']==1
    assert receipt['lookup_words']==probe.lookup_words()
    assert (BUILD/'mx8_warp_lut_generated.cuh').read_text()==probe.generated_header()
    assert (BUILD/'mx8_warp_lut_prepare.cu').read_text()==probe.generated_host()
    for name,digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['artifact_sha256'].items():
        if name.endswith('.so'):
            assert not (BUILD/name).exists()  # Binary stays in the checked local archive, not Git.
            continue
        assert hashlib.sha256((BUILD/name).read_bytes()).hexdigest()==digest


def test_source_identity_runtime_receipts_and_same_native_INT4_GEMM():
    env=read(DATA/'environment.json')
    assert env['git_commit']=='0f709ea8c7dee41a06d30bcfdc1ca664f7c32f36'
    assert env['extension_sha256']==EXTENSION_SHA
    assert env['no_filtering'] and not env['production_default_changed']
    assert env['source_quantization']=='original_FP16_direct_source_quantization_excluded'
    assert (env['args']['samples'],env['args']['rounds'],env['args']['warmup'],
            env['args']['repeats'],env['args']['inner'])==(24,3,50,200,100)
    assert env['codegen']['conversion_lookup']==read(BUILD/'build.json')
    assert env['codegen']['runtime_source_sha256']==hashlib.sha256(
        (ROOT/'scripts/benchmark_o7_warp_lut.py').read_bytes()).hexdigest()
    measured=[env['resources'][str(p)] for p in (0,1)]
    assert measured[0]['preparation_implementation']=='v73_row_fused'
    assert measured[1]['preparation_implementation']=='MXFP8_warp_lookup'
    for r in measured:
        assert r['kernel_symbol']=='adangel_roof_o78_eight_chain_candidate'
        assert r['cta_tile']==[64,128,128] and r['pipeline_stages']==2
        assert r['registers_per_thread']==168 and r['active_blocks_per_sm']==3
        assert r['local_size_bytes']==0 and not r['GEMM_modified']
    provenance=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    assert hashlib.sha256(provenance.read_bytes()).hexdigest()==read(DATA/'source_identity_checked.json')['old_provenance_sha256']
    old={r['sample_id']:r for r in map(json.loads,provenance.read_text().splitlines()) if r['variant']=='o7'}
    rows=list(map(json.loads,(DATA/'source_provenance.jsonl').read_text().splitlines()))
    assert len(rows)==24 and {r['sample_id'] for r in rows}==set(old)
    assert all(r==old[r['sample_id']] for r in rows)
    # Reuse the audited v78 cubin, not a conversion probe as an MMA audit.
    sass=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    function=entries(sass.read_text())['adangel_roof_o78_eight_chain_candidate']
    assert any('IMMA' in op and 'U4.S4' in op for op in function['opcode_counts'])
    assert any('IMMA' in op and 'S4.S4' in op for op in function['opcode_counts'])


def test_exhaustive_correctness_and_limited_sanitizer_scope():
    for run in ('preflight','memcheck','synccheck','full24'):
        validation=read(BASE/f'runs/o378_roof_v106_{run}/validation.json')
        assert validation['passed'] and validation['count']==32 and validation['edge_count']==9
        lookup=validation['warp_lookup_exhaustive']
        assert lookup['passed'] and lookup['thread_code_combinations']==65536
        assert lookup['valid_E4M3FN_codes']==254 and lookup['invalid_127_255_not_admitted_as_sources']
        assert lookup['nondefault_stream']
    for tool in ('memcheck','synccheck'):
        log=(BASE/f'reports/o378_roof_v106_{tool}.log').read_text()
        assert 'VALIDATION PASSED' in log and 'ERROR SUMMARY: 0 errors' in log
    # This is finite small-M/N, full-K4096 and a conversion-entry filter,
    # not a full4096cubed / all-kernel racecheck acceptance claim.
    assert 'small_MN' in read(BASE/'runs/o378_roof_v106_preflight/validation.json')['scope']

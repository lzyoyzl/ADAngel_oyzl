import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from freeze_o3_dp2a_runtime_evidence import analyze,RUN,EXPECTED_EXTENSION
from benchmark_o3_dp2a import CUBIN_SHA,reviewed

BASE=ROOT/'docs/evidence/a100_o378_roof_v135'


def test_all_frozen_bytes_and_historical_runtime_sources():
    index=json.loads((BASE/'index.json').read_text())
    assert index['artifact_count']==len(index['files'])==42
    assert index['artifact_bytes']==sum(r['bytes'] for r in index['files'])
    for r in index['files']:
        data=(BASE/r['path']).read_bytes()
        assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
        # Nested build/ evidence is intentionally tracked despite the repo's
        # generic build ignore; a local-only fixture is not reproducible.
        subprocess.check_output(['git','ls-files','--error-unmatch','--',
            str((BASE/r['path']).relative_to(ROOT))],cwd=ROOT)
    assert hashlib.sha256((BASE/'analysis.json').read_bytes()).hexdigest()==index['analysis_sha256']
    build=json.loads((BASE/RUN/'build/dp2a_build.json').read_text())
    for path,sha in build['runtime_sources'].items():
        source=subprocess.check_output(['git','show',index['source_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==sha


def test_rebuild_all144_rows_without_filtering_or_gpu():
    result=analyze(BASE)
    assert result==json.loads((BASE/'analysis.json').read_text())
    assert result['records']==144 and result['raw_cuda_event_durations']==57600
    assert result['control_cv_failed']==2 and result['candidate_cv_failed']==0
    assert result['paired_speedup_ci95'][1]<1
    assert not result['four_mode_expansion_justified']
    assert result['bitwise_best_all'] and result['mse_vs_best']==0.0
    assert result['mean_dp2a_cta_fraction']==0.8072916666666666
    assert all(r['paired_throughput_change_percent']<0 for r in result['per_round'])


def test_original_gate_retained_native_int4_and_same_cubin():
    r=json.loads((BASE/RUN/'build/dp2a_build.json').read_text())
    assert reviewed(r['codegen'])==r['review']
    assert not r['review']['original_gate']['passed']
    assert r['codegen']['cubin_sha256']==CUBIN_SHA and r['no_GEMM_recompile']
    assert r['candidate_weight_conversion_includes_metadata_pack']
    final=(BASE/'reports/o378_v135_final_checks.log').read_text()
    assert EXPECTED_EXTENSION in final and CUBIN_SHA in final and '22 passed' in final
    assert not list(BASE.rglob('*.so')) and not list(BASE.rglob('*.cubin'))


def test_failed_test_harness_runs_preserved_not_counted_as_performance():
    assert 'AssertionError' in (BASE/'reports/o378_v135_runtime.log').read_text()
    assert 'mat1 and mat2 shapes cannot be multiplied' in (BASE/'reports/o378_v135_contract_runtime.log').read_text()
    assert not (BASE/'runs/o378_roof_v135_full24/results.jsonl').exists()
    assert not (BASE/'runs/o378_roof_v135_full24_contract/results.jsonl').exists()
    r=json.loads((BASE/'analysis.json').read_text())
    assert not r['production_default_changed'] and not r['new_NCU'] and not r['new_sanitizer']
    assert r['numerical_validation_cases']==96 and r['rejected_invalid_cases']==8
    assert r['extra_grouped_cases']==2 and r['extra_three_path_cases']==4

"""Frozen v137 evidence: hashes, all timings, math/ISA and safety boundaries."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'python'))
from freeze_o3_route_cohort_evidence import analyze

BASE=ROOT/'docs/evidence/a100_o378_roof_v137'


def test_all_frozen_file_hashes():
    index=json.loads((BASE/'index.json').read_text())
    paths=set()
    for row in index['files']:
        path=BASE/row['path'];data=path.read_bytes()
        assert row['path'] not in paths;paths.add(row['path'])
        assert len(data)==row['bytes']
        assert hashlib.sha256(data).hexdigest()==row['sha256']
        subprocess.check_output(['git','ls-files','--error-unmatch','--',str(path.relative_to(ROOT))],cwd=ROOT)
    assert len(paths)==index['artifact_count']
    assert sum(row['bytes'] for row in index['files'])==index['artifact_bytes']
    assert hashlib.sha256((BASE/'analysis.json').read_bytes()).hexdigest()==index['analysis_sha256']


def test_full24_statistics_safety_and_isa_replay():
    actual=analyze(BASE)
    assert actual==json.loads((BASE/'analysis.json').read_text())
    assert actual['records']==144 and actual['all_real_integer_path']
    assert actual['mse_vs_best']==0 and actual['bitwise_best_all']
    assert actual['resources']['v137']['threads']==256
    assert actual['resources']['v137']['active_blocks_per_sm']==2
    assert actual['resources']['v89']['active_blocks_per_sm']==3
    assert actual['integer_loop_local_loads']==actual['integer_loop_local_stores']==0
    # Cold fallback has spill even though the guarded main path does not.
    assert actual['resources']['v137']['local_size_bytes']>0
    assert actual['numerical_validation_cases']==96
    assert actual['rejected_invalid_cases']==8 and actual['extra_grouped_cases']==2
    assert actual['production_default_changed'] is False
    assert actual['paired_speedup_ci95'][1]<1 and not actual['four_mode_expansion_justified']
    assert actual['control_cv_failed']==actual['candidate_cv_failed']==0


def test_runtime_source_provenance_and_no_binaries_in_git_evidence():
    index=json.loads((BASE/'index.json').read_text())
    record=json.loads((BASE/'runs/o378_roof_v137_full24/build/cohort_build.json').read_text())
    for path,digest in record['runtime_sources'].items():
        source=subprocess.check_output(['git','show',index['source_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==digest
    assert record['no_GEMM_recompile'] and record['preparation_identical']
    assert not list(BASE.rglob('*.so')) and not list(BASE.rglob('*.cubin'))


def test_original_compile_failure_preserved():
    initial=(BASE/'reports/o378_roof_v137_o3_route_cohort_codegen/build.log').read_text()
    assert 'error:' in initial and 'prefetch' in initial
    fixed=json.loads((BASE/'reports/o378_roof_v137_o3_route_cohort_fixed_codegen/codegen.json').read_text())
    assert fixed['control_comparison']['passed'] and fixed['cost_gate']['passed']

"""Replay exact v98 runtime evidence; never turn a failed heuristic into a pass."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
from validate_unmeasured_eight_warp import summarize,resources_gate
from benchmark_a100_o1 import stats

BASE=ROOT/'docs/evidence/a100_o378_roof_v104'
DATA=BASE/'runs/o378_roof_v104_full24_r2'


def rows():
    return list(map(json.loads,(DATA/'results.jsonl').read_text().splitlines()))


def test_full24_raw_timings_and_summary_reproduce():
    records=rows()
    assert len(records)==432
    assert summarize(records)==json.loads((DATA/'summary.json').read_text())
    for row in records:
        assert len(row['raw_ms'])==200 and min(row['raw_ms'])>0
        assert row['summary']==pytest.approx(stats(row['raw_ms']),rel=1e-12,abs=1e-12)
        assert row['finite_fp32'] and row['bitwise_equal_control'] and row['MSE_regression_passed']
        assert not row['conversion_included'] and row['original_v98_static_gate_unchanged']
        assert row['mode']=='compute_only'
    summary=summarize(records)
    assert not summary['production_default_changed'] and not summary['original_v98_compile_gate_changed']
    assert summary['no_filtering']
    assert all(r['paired_ci95'][1]<1 for r in summary['summary'] if r['policy']==1)


def test_all_outputs_and_MSE_remain_identical():
    grouped={}
    for row in rows():
        grouped.setdefault((row['sample_id'],row['variant']),set()).add(row['mse_vs_reference'])
        assert row['reference']=={'o3':'o0','o7':'o5','o8':'o6'}[row['variant']]
    assert len(grouped)==72 and all(len(v)==1 for v in grouped.values())
    validation=json.loads((DATA/'validation.json').read_text())
    assert validation['passed'] and validation['count']==12
    assert {(r['variant'],r['pattern']) for r in validation['checks']}=={
        (v,p) for v in ('o3','o7','o8') for p in ('random','zero','saturated','fallback')}
    assert all(r['nondefault_stream'] and r['finite_fp32'] and r['bitwise_current_best']
               and r['production_semantics_close'] for r in validation['checks'])


def test_existing_sources_cubins_and_actual_capacity():
    env=json.loads((DATA/'environment.json').read_text())
    assert env['modes']==['compute_only'] and env['conversion_excluded']
    assert env['metadata_prepared_once_shared'] and not env['native_extension_rebuilt']
    assert not env['production_default_changed'] and env['no_filtering']
    assert env['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    for name,digest in env['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    resources={(kind,p):env['resources'][f'{kind}_{p}'] for kind in ('o3','o78') for p in (0,1)}
    assert resources_gate(resources)
    for kind in ('o3','o78'):
        old,new=(resources[kind,p] for p in (0,1))
        assert (old['active_warps_per_sm'],new['active_warps_per_sm'])==(12,16)
        assert new['local_size_bytes']==0
        directory=ROOT/f'docs/evidence/a100_o378_roof_v98/reports/o378_roof_v98_{kind}_codegen'
        stem={'o3':'o3_eight_warp_fullk','o78':'o78_eight_warp_fullk'}[kind]
        assert hashlib.sha256((directory/(stem+'.cubin')).read_bytes()).hexdigest()==new['cubin_sha256']
    assert env['rationale']['v98_original_heuristic_failed']
    assert not env['rationale']['v98_original_compile_gate_changed']


def test_same_full_source_provenance_and_preserved_failures():
    old_path=ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl'
    env=json.loads((DATA/'environment.json').read_text())
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==env['source_provenance_sha256']
    old={(r['sample_id'],r['variant']):r for r in map(json.loads,old_path.read_text().splitlines())}
    new=list(map(json.loads,(DATA/'source_provenance.jsonl').read_text().splitlines()))
    assert len(new)==48 and all(r==old[r['sample_id'],r['variant']] for r in new)
    reports=BASE/'reports'
    assert 'Torch CUDA context must be current' in (reports/'o378_roof_v104_resources.log').read_text()
    assert "KeyError: 'packed_activation_g128_major'" in (reports/'o378_roof_v104_full24.log').read_text()
    assert '14 passed' in (reports/'o378_roof_v104_unit_tests.log').read_text()
    assert (reports/'o378_roof_v104_full24_r2.log').read_text().count('full24 paired progress')==72

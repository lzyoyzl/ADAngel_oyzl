"""Replay full24 opportunity evidence on CPU; do not claim measured speedup."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_runtime_factor_reuse import factor_observation,summarize,PROVENANCE
from o78_fullk_integer_metadata import operand_metadata
from benchmark_o78_fullk_gpu_prepare import expected_operand_gpu

BASE=ROOT/'docs/evidence/a100_o378_roof_v105/reports'
DATA=BASE/'o378_roof_v105_runtime_factor_r2'


def rows():
    return list(map(json.loads,(DATA/'results.jsonl').read_text().splitlines()))


def test_complete_opportunity_summary_and_stop_decision():
    result=summarize(rows())
    assert result==json.loads((DATA/'summary.json').read_text())
    assert not any(result[k] for k in ('new_MSE_measured','new_GEMM_measured',
                                     'new_kernel_implemented','production_default_changed'))
    assert result['O3_unchanged'] and result['no_quantization_or_scale_change']
    assert all(not v['raw_opportunity_gate'] for v in result['variants'])
    assert [v['mean_same4_optimistic_loop_work_reduction_percent'] for v in result['variants']]==pytest.approx(
        [.07322093940382943,0],rel=1e-12,abs=1e-15)


def test_all48_factor_guards_and_SIMD_statistics_reproduce_from_scale_codes():
    for row in rows():
        path=DATA/row['artifact']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            assert arrays['scale_codes'].shape==(4096,32)
            cpu=operand_metadata(arrays['scale_codes'],arrays['group_sum_squares'],
                                 row['scale_kind'],arrays['base_multiplier'])
            actual=expected_operand_gpu(cpu,activation=True)
            assert np.array_equal(actual['factors'],arrays['factors'])
            assert np.array_equal(actual['row_status'],arrays['row_status'])
            assert hashlib.sha256(actual['factors'].tobytes()).hexdigest()==row['factors_sha256']
            s=factor_observation(actual['factors'],actual['row_status'])
        assert s==row['statistics']
        assert s['shape']==[32,4096] and s['warp_panels']==4096
        assert s['representable_warp_panels']+s['excluded_warp_panels']==s['warp_panels']
        assert sum(s['warp_nontrivial_unique_vector_histogram'])==s['representable_warp_panels']
        assert sum(s['per_lane_nontrivial_unique_histogram'])==8*s['representable_warp_panels']
        assert 'fallback ignored' in s['exclusions']


def test_original_source_identity_code_receipts_and_unchanged_extension():
    env=json.loads((DATA/'environment.json').read_text())
    for path,digest in env['source_hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    provenance=ROOT/PROVENANCE
    assert hashlib.sha256(provenance.read_bytes()).hexdigest()==env['provenance_sha256']
    old={(r['sample_id'],r['variant']):r for r in map(json.loads,provenance.read_text().splitlines())}
    for row in rows():
        r=old[row['sample_id'],row['variant']]
        assert row['source']==r['activation'] and row['raw_sha256']==r['raw_sha256']
        assert row['v99_source_exact']
    assert env['no_native_GEMM_or_candidate_launched'] and env['no_performance_timing'] and env['no_new_MSE']
    assert env['native_extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    receipt=env['coordinates']['existing_best_receipt']['build']
    for path,digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest


def test_actual_CuTe_coordinates_and_preserved_preflight_failure():
    coords=json.loads((DATA/'coordinates.json').read_text())
    env=json.loads((DATA/'environment.json').read_text())
    assert coords==env['coordinates']['result']
    assert coords['passed'] and not coords['gpu_execution']
    assert coords['outputs']==8192 and coords['rows_per_thread']==4
    assert coords['quad_offsets']==[0,8,32,40] and coords['warp_m_bases']==[0,16,0,16]
    assert coords['quad_bases_per_warp']==8 and coords['lanes_per_quad']==4
    first=(BASE/'o378_roof_v105_runtime_factor.log').read_text()
    assert 'Assertion' in first and 'first+16,first+24' in first
    assert 'runtime factor observation passed' not in first
    assert (BASE/'o378_roof_v105_runtime_factor_r2.log').read_text().count(
        'runtime factor observation passed')==48

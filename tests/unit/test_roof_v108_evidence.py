"""CPU replay of v108 opportunity data; no new GPU speed/MSE assertion."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_scale_profile_buckets import (
    OBSERVATIONS,PROVENANCE,activation_profiles,constant_weights,
    observation_authority,summarize,
)

DATA=ROOT/'docs/evidence/a100_o378_roof_v108/reports/o378_roof_v108_profile_buckets'
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def rows():
    return list(map(json.loads,(DATA/'results.jsonl').read_text().splitlines()))


def test_full24_original_source_identity_and_no_GPU_work_receipts():
    r=rows();ids={x['sample_id'] for x in r}
    expected={f'layer_{layer:02d}_{proj}_proj' for layer in (0,6,12,18,24,31) for proj in ('q','k','v','o')}
    assert ids==expected and len(r)==72
    env=json.loads((DATA/'environment.json').read_text())
    prior,authority=observation_authority(ROOT,ids,digest)
    assert env['git_commit']=='5f5f8a877fbe050ac1d63dd9e7b49777110fc5fc'
    assert env['script_sha256']==digest(ROOT/'scripts/inspect_scale_profile_buckets.py')
    assert env['v105_results_sha256']==digest(OBSERVATIONS/'results.jsonl')
    assert env['v105_environment_sha256']==digest(OBSERVATIONS/'environment.json')
    assert env['source_provenance_sha256']==digest(ROOT/PROVENANCE)
    assert env['prepared_manifest_sha256']==prior['prepared_manifest_sha256']
    assert env['native_extension_sha256']==prior['native_extension_sha256']==\
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['no_GPU_kernel_launched'] and env['no_new_quantization']
    for row in r:
        if row['variant']!='o3':
            original=authority[row['sample_id'],row['variant']]
            assert row['source']==original['source'] and row['raw_sha256']==original['raw_sha256']


def test_all72_observations_replay_and_verify_artifact_hashes():
    for row in rows():
        path=(DATA/row['artifact']) if row['variant']=='o3' else ROOT/row['artifact']
        assert digest(path)==row['artifact_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            if row['variant']=='o3':
                assert set(arrays.files)=={'scale_codes'}
                c=arrays['scale_codes']
                assert c.shape==(4096,32) and c.dtype==np.uint8
                assert hashlib.sha256(c.tobytes()).hexdigest()==row['scale_codes_sha256']
                assert row['prepared_file']==row['sample_id']+'.pt'
                assert len(row['prepared_sha256'])==64
                result=constant_weights(c)
            else:
                assert arrays['factors'].shape==(32,4096)
                result=activation_profiles(arrays['factors'],arrays['row_status'])
        assert row['statistics']==result


def test_optimistic_work_gate_stops_all_variants_not_a_speedup_claim():
    result=summarize(rows())
    assert result==json.loads((DATA/'summary.json').read_text())
    assert [v['mean_covered_fraction'] for v in result['variants']]==pytest.approx([.125,.01953125,.01953125])
    assert [v['mean_optimistic_loop_work_reduction_percent'] for v in result['variants']]==pytest.approx(
        [2.476780185758514,.3263707571801566,.3263707571801566],rel=1e-12)
    assert all(not v['opportunity_gate'] for v in result['variants'])
    assert not any(result[k] for k in ('new_kernel_implemented','new_GEMM_measured',
                                     'new_MSE_measured','production_default_changed'))
    assert result['source_quantization_unchanged'] and result['no_filtering']


def test_preserved_single_completed_observation_run():
    log=(DATA.parent/'o378_roof_v108_profile_buckets.log').read_text()
    assert log.count('full-K scale-profile observation passed')==24
    assert 'Traceback' not in log

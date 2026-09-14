import itertools
import json
from pathlib import Path
import runpy

import pytest

VERIFY = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                           'scripts/verify_sm120_magic_results.py'))['verify']


@pytest.fixture
def formal_fixture(tmp_path):
    paired, four, safety = [tmp_path / x for x in ['paired', 'four', 'safety']]
    for p in [paired, four, safety]: p.mkdir()
    def write(p, x): p.write_text(json.dumps(x))
    env = dict(binary_sha256='test-binary', cuda_sources_sha256={'test.cu': 'hash'},
               torch='test', cuda='test', gpu='test')
    write(safety / 'environment.json', env)
    audit = tmp_path / 'audit.json'
    write(audit, dict(passed=True, binary_sha256='test-binary'))
    ids = [f'layer_{i:02d}_{p}' for i in [0, 6, 12, 18, 24, 31]
           for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj']]
    manifest = dict(matrix_shape=[4096]*3, samples=[dict(sample_id=i) for i in ids])
    for directory, modes, rounds in [
        (paired, ['compute_only'], 3),
        (four, ['conversion_only', 'compute_only', 'cold', 'steady_state'], 1),
    ]:
        write(directory/'environment.json', dict(env, args=dict(
            warmup=50, repeats=200, rounds=rounds, inner=100, samples=0, variants=['o1', 'o3'])))
        write(directory/'data_manifest.json', manifest)
        rows = []
        for sample, impl, mode, rnd in itertools.product(ids, ['o0', 'o1', 'o1_magic', 'o3', 'o3_magic'], modes, range(rounds)):
            variant = impl.split('_')[0]
            stages = {'total'}
            if mode != 'conversion_only': stages.add('gemm')
            if mode in ['conversion_only', 'cold']: stages.add('weight_conversion')
            if mode != 'compute_only' and variant != 'o1': stages.add('activation_conversion')
            rows.append(dict(sample_id=sample, implementation=impl, mode=mode, round=rnd,
                timings_ms={s: [.1]*200 for s in stages},
                summary={s: dict(cv_percent=0) for s in stages},
                timing_method=dict(strategy='conversion_amortized_end_to_end_direct', conversion_inner_repeats=100,
                    end_to_end_total_timing='direct_single_path', mode_total_inner_repeats=100 if mode=='conversion_only' else 1),
                mse_vs_o0=0 if variant=='o0' else 1, baseline_mse_vs_o0=1, bitwise_equal_baseline=True,
                kernel=dict(scale_publication='per_writer_acquire_release_v3', full_barrier_arrivals_per_stage=33,
                    integer_conversion='exact_magic_bias' if impl.endswith('_magic') else 'i2f',
                    cta_tile=[128,64,64] if variant=='o1' else [64,32,128], output_dtype='fp32', data_movement='TMA')))
        (directory/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return paired, four, safety, audit


def test_complete_formal_magic_results(formal_fixture):
    result = VERIFY(*formal_fixture)
    assert result['passed']
    assert [r['records'] for r in result['runs']] == [360, 480]


def test_explicit_paired_only_does_not_certify_four_modes(formal_fixture):
    paired, _, safety, audit = formal_fixture
    result = VERIFY(paired, None, safety, audit)
    assert result['passed'] and result['scope'] == 'paired_only'
    assert [r['records'] for r in result['runs']] == [360]


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'short_timing', 'old_sync', 'mse'])
def test_formal_magic_results_reject_corruption(formal_fixture, corruption):
    p = formal_fixture[0] / 'results.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    r = next(r for r in rows if r['implementation']=='o1_magic')
    if corruption=='duplicate': rows.append(rows[0])
    if corruption=='missing': rows.pop()
    if corruption=='short_timing': r['timings_ms']['gemm'].pop()
    if corruption=='old_sync': r['kernel']['scale_publication']='unfixed'
    if corruption=='mse': r['mse_vs_o0']=2
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(AssertionError): VERIFY(*formal_fixture)

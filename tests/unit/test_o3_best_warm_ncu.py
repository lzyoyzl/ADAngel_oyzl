"""CPU-only cache-primed profiler provenance contracts, not GPU verification."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from profile_o3_best_warm import SAMPLE, SYMBOL, WARMUP, select_entry
from run_o3_best_warm_ncu import capture_command, verify_passes


def receipt():
    return dict(sample_id=SAMPLE, variant='o3', shape=[4096]*3, policy=1, expected_kernel=SYMBOL,
        gemm_binary_sha256='a'*64, extension_sha256_before='b'*64, extension_sha256_after='b'*64,
        resources={'registers_per_thread': 168}, guard=dict(integer_ctas=2048, fallback_ctas=0, invalid_ctas=0),
        validation_sha256='c'*64, git_commit='d'*40, source_sha256='e'*64,
        raw_manifest_sha256='f'*64, prepared_manifest_sha256='1'*64,
        raw_sample_sha256='2'*64, prepared_sample_sha256='3'*64,
        numerical_checks_passed=True, bitwise_previous_best=True, finite_fp32=True,
        mse_vs_previous_best=0, replay_mode='application', cache_control='none', clock_control='none',
        filtered_launch_skip=WARMUP, filtered_launch_count=1, production_default_changed=False,
        new_performance_result=False, new_candidate_implemented=False)


def test_capture_replays_warmups_without_cache_flush_or_clock_changes():
    cmd = capture_command('ncu', 'python', 'prefix', 'validation', 'receipts')
    for flag, val in (('--set', 'full'), ('--replay-mode', 'application'),
            ('--cache-control', 'none'), ('--clock-control', 'none'),
            ('--launch-skip', str(WARMUP)), ('--launch-count', '1')):
        assert cmd[cmd.index(flag)+1] == val
    assert cmd[cmd.index('--kernel-name')+1] == 'regex:^'+SYMBOL+'$'
    assert '--force-overwrite' not in cmd
    assert cmd[cmd.index('--output')+1] == 'receipts'


def test_exact_sample_and_every_replay_pass():
    one = receipt(); second = deepcopy(one); second['pid'] = 2
    assert verify_passes([one, second]) == one
    assert select_entry(dict(samples=[dict(sample_id=SAMPLE)]))['sample_id'] == SAMPLE
    with pytest.raises(ValueError): verify_passes([])
    with pytest.raises(ValueError): select_entry(dict(samples=[]))
    bad = deepcopy(one); bad['prepared_sample_sha256'] = '4'*64
    with pytest.raises(ValueError): verify_passes([one, bad, second])


@pytest.mark.parametrize('key,value', [('cache_control', 'all'), ('clock_control', 'base'),
    ('production_default_changed', True), ('new_performance_result', True), ('policy', 0),
    ('numerical_checks_passed', False), ('mse_vs_previous_best', 1e-8), ('filtered_launch_skip', 50)])
def test_nonconforming_capture_not_accepted(key, value):
    bad = receipt(); bad[key] = value
    with pytest.raises(ValueError): verify_passes([bad])

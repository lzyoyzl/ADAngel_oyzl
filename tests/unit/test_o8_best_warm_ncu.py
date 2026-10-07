"""CPU profiler provenance contracts; not GPU or Event performance tests."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from profile_o8_best_warm import SAMPLE, SYMBOL, select_entry
from run_o8_best_warm_ncu import capture_command, verify_passes


def test_cache_priming_replayed_application_without_clock_change_or_extra_launches():
    cmd = capture_command('ncu', 'python', 'report', 'validation', 'receipts')
    for flag, val in (('--set', 'full'), ('--replay-mode', 'application'),
                      ('--cache-control', 'none'), ('--clock-control', 'none'),
                      ('--launch-skip', '50'), ('--launch-count', '1')):
        assert cmd[cmd.index(flag)+1] == val
    assert cmd[cmd.index('--kernel-name')+1] == 'regex:^'+SYMBOL+'$'
    assert '--force-overwrite' not in cmd
    assert cmd[cmd.index('--output')+1] == 'receipts'


def test_selected_real_sample_cannot_silently_fall_back_to_first_layer():
    manifest = dict(samples=[dict(sample_id='layer_00_q_proj'), dict(sample_id=SAMPLE)])
    assert select_entry(manifest) == manifest['samples'][1]
    with pytest.raises(ValueError): select_entry(dict(samples=manifest['samples'][:1]))
    with pytest.raises(ValueError): select_entry(dict(samples=manifest['samples'][1:]*2))


def receipt():
    return dict(sample_id=SAMPLE, variant='o8', shape=[4096]*3, expected_kernel=SYMBOL,
        source_identity={'exact': True}, gemm_binary_sha256='a'*64,
        extension_sha256_before='b'*64, extension_sha256_after='b'*64, resources={'registers': 168},
        guard={'integer_ctas': 2048, 'fallback_ctas': 0, 'invalid_ctas': 0},
        validation_sha256='c'*64, git_commit='d'*40, source_sha256='e'*64,
        numerical_checks_passed=True, bitwise_previous_fullK=True, finite_fp32=True,
        mse_vs_previous_fullK=0, replay_mode='application', cache_control='none', clock_control='none',
        filtered_launch_skip=50, filtered_launch_count=1, production_default_changed=False,
        new_GEMM_measured=False, new_candidate_implemented=False)


def test_all_completed_pass_receipts_checked_not_last_pass_only():
    one = receipt(); other = deepcopy(one); other['pid'] = 2
    assert verify_passes([one, other]) == one
    bad = deepcopy(one); bad['gemm_binary_sha256'] = 'f'*64
    with pytest.raises(ValueError): verify_passes([one, bad, other])
    with pytest.raises(ValueError): verify_passes([])


@pytest.mark.parametrize('key,value', [('cache_control', 'all'), ('clock_control', 'base'),
    ('production_default_changed', True), ('new_GEMM_measured', True),
    ('numerical_checks_passed', False), ('mse_vs_previous_fullK', 1e-8)])
def test_invalid_diagnostic_contract_not_accepted_as_same_best_or_new_speedup(key, value):
    bad = receipt(); bad[key] = value
    with pytest.raises(ValueError): verify_passes([bad])

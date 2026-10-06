import copy
from pathlib import Path
import statistics
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_grouped_cta_retest import SAMPLES, compare, summarize


def rows():
    result = []
    for sid in sorted(SAMPLES):
        for impl in (0, 1):
            values = [1.0, 1.0, 2.0] if impl == 0 else [.9, .9, 1.8]
            mean = statistics.fmean(values)
            stage = dict(median_ms=statistics.median(values), mean_ms=mean,
                         cv_percent=statistics.pstdev(values) / mean * 100)
            result.append(dict(sample_id=sid, variant='o3', mode='compute_only', round=0,
                implementation=impl, raw_ms=dict(gemm=values), stage_summaries=dict(gemm=stage),
                summary=stage, bitwise_equal_current_best=True, payload_bitwise=True,
                mse_vs_current_best=0, mse_vs_paired_fp16=.01))
    return result


def test_unfiltered_pair_summary_and_mse():
    result = summarize(rows(), 1, 3)
    assert len(result) == 2
    assert result[0]['any_stage_cv_failed_records'] == result[1]['any_stage_cv_failed_records'] == 24
    assert result[1]['paired_speedup'] == pytest.approx(1 / .9)
    assert result[1]['median_mse'] == .01
    assert result[1]['samples_faster_than_control'] == 24


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'raw_count', 'raw_stat', 'mse', 'output'])
def test_rejects_invalid_evidence(change):
    evidence = rows()
    if change == 'missing':
        evidence.pop()
    elif change == 'duplicate':
        evidence.append(copy.deepcopy(evidence[0]))
    elif change == 'raw_count':
        evidence[0]['raw_ms']['gemm'].pop()
    elif change == 'raw_stat':
        evidence[0]['stage_summaries']['gemm']['cv_percent'] = 0
    elif change == 'mse':
        evidence[0]['mse_vs_paired_fp16'] = .02
    elif change == 'output':
        evidence[0]['mse_vs_current_best'] = .01
    with pytest.raises(ValueError):
        summarize(evidence, 1, 3)


def test_only_warmup_change_and_not_false_stability():
    first = dict(summary=summarize(rows(), 1, 3), fingerprint=dict(binary='same'),
                 args=dict(samples=24, rounds=1, repeats=3, inner=100, warmup=50))
    retry = copy.deepcopy(first)
    retry['args']['warmup'] = 1000
    result = compare(first, retry)
    assert all(not r['strict_cv_passed'] for r in result)
    retry['fingerprint']['binary'] = 'different'
    with pytest.raises(ValueError):
        compare(first, retry)

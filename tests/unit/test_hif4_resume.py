"""Crash recovery must select by completeness, never favorable measurements."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from resume_hif4_swar import MODES, completed_prefix


def fixture_rows(n):
    ids = [f's{i}' for i in range(24)]
    rows = [dict(sample_id=s, variant='o8', mode=m, round=r, candidate=p,
                 summary={'cv_percent': 99, 'median_ms': 99})
            for s in ids for r in range(3) for m in MODES for p in (0, 1)]
    return ids, rows[:n]


def test_interrupted_partial_is_retained_outside_complete_dataset():
    ids, rows = fixture_rows(483)
    start, kept, partial = completed_prefix(rows, ids)
    assert start == 20 and kept == rows[:480] and partial == rows[480:]
    assert all(r['summary']['cv_percent'] == 99 for r in kept)
    for r in rows:
        r['summary']['cv_percent'] = 0
        r['summary']['median_ms'] = .00001
    assert completed_prefix(rows, ids)[0] == 20


def test_reject_duplicates_unknown_holes_and_already_complete():
    ids, rows = fixture_rows(483)
    with pytest.raises(ValueError):
        completed_prefix(rows + [rows[0]], ids)
    with pytest.raises(ValueError):
        completed_prefix(rows[1:], ids)
    with pytest.raises(ValueError):
        completed_prefix(rows + [dict(rows[0], sample_id='unknown')], ids)
    ids, rows = fixture_rows(576)
    with pytest.raises(ValueError):
        completed_prefix(rows, ids)


def test_exact_boundary_and_empty_prefix():
    ids, rows = fixture_rows(480)
    assert completed_prefix(rows, ids) == (20, rows, [])
    assert completed_prefix([], ids) == (0, [], [])

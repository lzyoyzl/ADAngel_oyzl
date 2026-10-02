"""Exact input-range lower bound before spending GPU optimization time."""
from functools import reduce
import math
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_o78_input_factor_folding import inspect_operand
from o78_fullk_integer_metadata import dyadic_scale


def inspect(codes, payload, kind='ue8m0', bits=8, tile=1):
    q = np.asarray(payload, dtype=np.int16)
    return inspect_operand(np.asarray(codes), q.min(-1), q.max(-1), np.gcd.reduce(q, axis=-1), kind, bits, tile)


def test_extrema_zero_and_signed_edges():
    r = inspect([[127], [127], [127]], [[[-128, 127]], [[-128, 128]], [[0, 0]]])
    # Second row has a removable GCD of128, and hence exact [-1,+1].
    assert [x['fits'] for x in r['rows']] == [True, True, True]
    assert [x['minimum_signed_bits'] for x in r['rows']] == [8, 2, 1]
    r = inspect([[127, 128]], [[[1, 3], [-128, 127]]])
    assert not r['rows'][0]['fits'] and r['rows'][0]['minimum_signed_bits'] == 9


def test_zero_payload_does_not_force_a_worse_anchor():
    r = inspect([[0, 254]], [[[0, 0], [-7, 6]]], bits=4)
    assert r['rows'][0]['anchor'] == 127 and r['rows'][0]['fits']
    r = inspect([[0, 126]], [[[-127, 126], [-7, 6]]], kind='e4m3', bits=4)
    assert r['rows'][0]['fits']  # first group has exactly zero scale


def test_exact_stats_match_exhaustive_weighted_payload():
    rng = random.Random(20261003)
    for kind, limit in (('ue8m0', 254), ('e4m3', 126), ('e6m2', 254)):
        for _ in range(100):
            codes = [[rng.randrange(limit + 1) for _ in range(4)]]
            q = [[[rng.randrange(-128, 128) for _ in range(7)] for _ in range(4)]]
            got = inspect(codes, q, kind)['rows'][0]
            ds = [dyadic_scale(c, kind) for c in codes[0]]
            anchor = min((e for (m, e), g in zip(ds, q[0]) if m and any(g)), default=0)
            values = [v * (m << (e - anchor)) if m and any(g) else 0
                      for (m, e), g in zip(ds, q[0]) for v in g]
            divisor = reduce(math.gcd, values, 0) or 1
            assert (got['minimum'], got['maximum'], got['removed_gcd']) == (min(values) // divisor, max(values) // divisor, divisor)


def test_tile_gate_requires_every_row():
    q = [[[1, 3], [2, 4]], [[1, 3], [6, 7]], [[1, 3], [6, 7]], [[1, 3], [2, 4]]]
    r = inspect([[127, 128]] * 4, q, bits=4, tile=2)
    assert r['fitting_rows'] == 0  # even4*2 exceeds positive INT4 range
    r = inspect([[127, 127]] * 4, q, bits=4, tile=2)
    assert r['fitting_tiles'] == 2


@pytest.mark.parametrize('kind,code', [('ue8m0', 255), ('e4m3', 127), ('e6m2', 255)])
def test_invalid_source_codes_are_rejected(kind, code):
    with pytest.raises(ValueError):
        inspect([[code]], [[[0, 0]]], kind=kind)

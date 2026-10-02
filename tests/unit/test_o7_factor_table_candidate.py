"""v76 exact coefficients and generated-source/guard contracts; no GPU claims."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_o7_factor_table_codegen import table_header, TABLE_ROWS, SHARED_BYTES


def test_generator_has_only_scoped_changes_and_pipeline_contract():
    old = (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    new = table_header(old)
    assert SHARED_BYTES == 2 * (64 * 4 + (64 + 64 + 128) * 64 + TABLE_ROWS * 128 * 4)
    assert 'static_assert(sizeof(Storage)==43520)' in new
    assert 'uint32_t coefficients[2][10][128]' in new
    assert new.count('__syncthreads()') == old.count('__syncthreads()') == 1
    for marker in ('cp.async.commit_group;', 'cp.async.wait_group 0;', 'SM80_16x8x64_S32U4S4S32_TN',
                   'SM80_16x8x64_S32S4S4S32_TN', 'acc(vi,mi,full_ni)+=partial*coefficient;',
                   '// Reuse the 64 INT32 register slots as FP32 bits before any output store.'):
        assert marker in new
    assert new.split('// Reuse the 64 INT32')[1] == old.split('// Reuse the 64 INT32')[1].replace(
        'o78_fullk_integer_experiment', 'o7_factor_table_experiment')
    assert 's.weight_factors' not in new
    with pytest.raises(ValueError):
        table_header(old.replace('int weight_factors[2][128]', 'int wrong[2][128]'))


def test_table_exact_for_selected_guarded_coefficients():
    rng = random.Random(20261003)
    for d in range(TABLE_ROWS):
        for w in [0, 1, 7, ((1 << 31)-1) >> d] + [rng.randrange(1 << 31) for _ in range(500)]:
            table_bits = (w << d) & 0xffffffff
            if w * (1 << d) <= (1 << 31)-1:
                assert table_bits == w * (1 << d)
    # Non-power, zero, negative and over-width factors must use unchanged v67.
    fits = lambda v: 0 < v < 1024 and (v & (v-1)) == 0
    assert all(fits(1 << d) for d in range(10))
    assert not any(fits(v) for v in (0, -1, 3, 6, 1024, (1 << 31)-1))


def test_cta_guard_and_existing_fallbacks_are_preserved():
    text = (ROOT / 'csrc/sm80/roof_o7_factor_table_probe.cu').read_text()
    assert 'flag>1u' in text and 'flag==1u' in text
    assert 'item<32*64;item+=128' in text and '__syncthreads_or(bad)' in text
    assert 'o78_fullk_integer_experiment::body' in text
    assert 'o7_factor_table_experiment::body' in text
    assert 'f>=1024u' in text and '(f&(f-1u))!=0u' in text


def test_first_four_o7_input_factors_are_in_table_not_general_claim():
    directory = ROOT / 'docs/evidence/a100_o378_roof_v74/reports/o378_roof_v74_input_fold_screen'
    files = sorted(directory.glob('*_o7_a.json'))
    assert len(files) == 4
    for file in files:
        codes = np.asarray(json.loads(file.read_text())['codes'], dtype=np.int64)
        assert codes.shape == (4096, 32)
        assert (codes.max(1) - codes.min(1)).max() <= 9

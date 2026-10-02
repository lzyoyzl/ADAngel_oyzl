"""Recompute exact input-folding ranges from all retained per-group statistics."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_o78_input_factor_folding import inspect_operand

DATA = ROOT / 'docs/evidence/a100_o378_roof_v74/reports/o378_roof_v74_input_fold_screen'


def test_recompute_all_exact_ranges_and_source_identities():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = json.loads((DATA / 'summary.json').read_text())
    assert summary['scope'] == 'exact_input_factor_folding_feasibility_not_new_kernel_timing_or_MSE'
    assert summary['script_sha256'] == digest(ROOT / 'scripts/inspect_o78_input_factor_folding.py')
    expected = {(f'layer_00_{p}_proj', v) for p in ('q', 'k', 'v', 'o') for v in ('o7', 'o8')}
    assert len(summary['samples']) == len(expected) == 8
    assert {(r['sample_id'], r['variant']) for r in summary['samples']} == expected
    names = {'summary.json'}
    histograms = {(v, s): Counter() for v in ('o7', 'o8') for s in ('a', 'w')}
    for row in summary['samples']:
        assert row['ctas_fitting_both_operands'] == 0
        for side in ('a', 'w'):
            entry = row[side]
            names.add(entry['file'])
            assert digest(DATA / entry['file']) == entry['sha256']
            data = json.loads((DATA / entry['file']).read_text())
            arrays = [np.asarray(data[k], dtype=np.int64) for k in ('codes', 'minima', 'maxima', 'gcds')]
            assert all(x.shape == (4096, 32) for x in arrays)
            rebuilt = inspect_operand(*arrays, data['kind'], entry['target_bits'], entry['tile_rows'])
            # JSON serializes integer histogram keys as strings.
            rebuilt = json.loads(json.dumps(rebuilt))
            assert rebuilt == data['analysis']
            assert {k: v for k, v in rebuilt.items() if k not in ('rows', 'tile_mask')} == {
                k: v for k, v in entry.items() if k not in ('file', 'sha256')}
            assert rebuilt['fitting_rows'] == rebuilt['fitting_tiles'] == 0
            assert rebuilt['total_rows'] == 4096
            histograms[row['variant'], side].update({int(k): v for k, v in rebuilt['minimum_signed_bits_histogram'].items()})
    assert {p.name for p in DATA.iterdir()} == names
    assert dict(histograms['o7', 'a']) == {11: 12, 12: 1763, 13: 10258, 14: 3796, 15: 261, 17: 294}
    assert dict(histograms['o7', 'w']) == {8: 1888, 9: 6189, 10: 6691, 11: 1424, 12: 184, 13: 8}
    assert dict(histograms['o8', 'a']) == {13: 247, 14: 5194, 15: 6313, 16: 3897, 17: 439, 18: 294}
    assert dict(histograms['o8', 'w']) == {7: 2352, 8: 4097, 9: 7677, 10: 1990, 11: 250, 12: 18}

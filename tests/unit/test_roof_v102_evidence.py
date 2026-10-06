import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_small_atom_capacity import audit,summarize,SYMBOLS

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v102/reports'
FIXED=EVIDENCE/'o378_roof_v102_capacity_r2'


def test_original_work_mismatch_stops_before_timing():
    failed=EVIDENCE/'o378_roof_v102_capacity'
    with pytest.raises(ValueError,match='exact runtime group loop missing'):
        audit((failed/'kernel.sass').read_text())
    assert not (failed/'results.jsonl').exists()
    assert 'ValueError: exact runtime group loop missing' in (EVIDENCE/'first_failure.log').read_text()


def test_raw_paired_diagnostic_reproduces_summary_not_case_speedup():
    rows=[json.loads(s) for s in (FIXED/'results.jsonl').read_text().splitlines()]
    actual=summarize(rows)
    assert actual==json.loads((FIXED/'summary.json').read_text())
    assert actual['paired_bootstrap_95_ci'][1]<1
    assert not actual['full_GEMM_worth_implementing']
    assert not actual['new_experiment_MSE_measured']
    assert all(r['cv_passed'] for r in actual['records'])
    assert len(rows)==2 and all(len(r['raw_ms'])==200 for r in rows)


def test_actual_native_entries_and_tracked_text_hashes():
    receipt=json.loads((FIXED/'codegen.json').read_text())
    assert audit((FIXED/'kernel.sass').read_text())==receipt['entries']
    assert receipt['entries'][SYMBOLS[0]]['static_native_mma_count']==64
    assert receipt['entries'][SYMBOLS[1]]['static_native_mma_count']==256
    for name,expected in receipt['artifact_sha256'].items():
        if name=='small_atom_capacity':continue # retained in server/projecttmp binary archive
        assert hashlib.sha256((FIXED/name).read_bytes()).hexdigest()==expected
    for name,expected in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected

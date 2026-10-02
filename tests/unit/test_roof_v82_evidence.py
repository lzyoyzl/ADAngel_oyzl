"""Archived capacity diagnostic is not a new GEMM or quantization result."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_mma_issue_capacity import analyze
from probe_a100_mma_issue_capacity import SYMBOLS, audit

E = ROOT / 'docs/evidence/a100_o378_roof_v82/reports/o378_roof_v82_capacity_checked'


def test_event_ncu_and_real_machine_work_agree():
    got = analyze(E)
    assert got['static_and_dynamic_work_verified']
    assert not got['production_default_changed']
    assert got['no_new_trace_MSE_or_conversion_or_end_to_end']
    assert got['ncu']['physical_int4_operations'] == 2199023255552
    assert got['ncu']['dynamic_warp_instructions']['IMMA'] == 134217728
    assert sum(got['ncu']['dynamic_warp_instructions'].values()) == 327401472
    assert got['ncu']['duration_ms'] == 2.019712
    assert got['ncu']['tensor_active_percent'] == 87.297357
    assert [r['statistics']['median_ms'] for r in got['event_records']] == [1.79200006, 1.79200006, 2.0223999]
    assert all(r['statistics']['count'] == 200 and r['cv_passed'] for r in got['event_records'])
    for path, digest in json.loads((E / 'codegen.json').read_text())['sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_cold_pointer_spill_not_misrepresented_as_hot_loop():
    sass = (E / 'kernel.sass').read_text()
    entries = audit(sass)
    assert [entries[s]['cold_local_instructions'] for s in SYMBOLS] == [0, 0, 2]
    assert all(entries[s]['hot_loop_memory_instructions'] == 0 for s in SYMBOLS)
    # Eight declared chains per slice do not force eight simultaneously live
    # SASS chains. This diagnostic compiler schedule reaches only five.
    assert entries[SYMBOLS[2]]['merged_chain_trace']['peak_started_not_finished_chains'] == 5


def test_auditor_rejects_hidden_memory_in_mma_loop():
    from test_mma_issue_capacity import fake_sass
    # Insert a real local load in the same backward branch region, without
    # removing any of the 64 required native MMA instructions.
    broken = fake_sass().replace('/*0400*/ @P0 BRA 0x0 ;',
        '/*0400*/ LDL R20, [R1] ;\n/*0410*/ @P0 BRA 0x0 ;', 1)
    with pytest.raises(ValueError, match='memory access'):
        audit(broken)

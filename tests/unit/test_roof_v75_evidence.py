"""Reproduce v75 from original nvdisasm output, without requiring a GPU."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from inspect_o78_register_liveness import analyze


def test_original_liveness_and_baseline_identity():
    path = ROOT / 'docs/evidence/a100_o378_roof_v75/reports/o378_roof_v75_liveness'
    receipt = json.loads((path / 'analysis.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(path / 'o78_fullk_liveness.txt') == receipt['liveness_sha256']
    assert digest(ROOT / 'scripts/inspect_o78_register_liveness.py') == receipt['script_sha256']
    baseline = json.loads((ROOT / 'docs/evidence/a100_o378_roof_v67/reports/o378_roof_v67_codegen/codegen.json').read_text())
    assert receipt['baseline_cubin_sha256'] == baseline['cubin_sha256']
    assert not any(receipt[k] for k in ('default_changed', 'new_kernel_built', 'new_runtime_measurements'))
    result = analyze((path / 'o78_fullk_liveness.txt').read_text())
    assert all(receipt[k] == v for k, v in result.items())
    assert result['allocated_gpr'] == 168 and result['function_max_live_gpr'] == 165
    loops = {r['kind']: r for r in result['loops']}
    assert loops['integer']['max_live_gpr'] == 160
    assert loops['fp32_fallback']['max_live_gpr'] == 165
    assert loops['integer']['static_instructions'] == 378
    assert loops['fp32_fallback']['static_instructions'] == 435
    assert not any(op.startswith(('LDL', 'STL')) for op in loops['integer']['opcode_counts'])
    assert loops['fp32_fallback']['opcode_counts']['LDL'] == 2
    for row in loops.values():
        assert row['opcode_counts']['IMMA.16864.S4.S4'] == 32
        assert row['opcode_counts']['IMMA.16864.U4.S4'] == 32
    # A necessary register capacity condition, not a throughput prediction.
    assert result['register_only_four_cta_limit_per_thread'] == 128 < loops['integer']['max_live_gpr']

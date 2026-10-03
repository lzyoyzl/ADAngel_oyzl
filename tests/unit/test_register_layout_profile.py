"""The negative-layout diagnostic cannot become a benchmark or change CUDA."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_register_layout_ncu import analyze_capture
from profile_register_layout_kernel import SYMBOLS


def test_existing_best_capture_is_accepted_only_with_exact_identity():
    e=ROOT/'docs/evidence/a100_o378_roof_v80_ncu/reports/o378_roof_v80_ncu'
    receipt=json.loads((e/'o7/receipt.json').read_text())
    receipt.update(policy=0,packed_candidate_verified=True)
    raw=(e/'o7_raw.csv').read_text(encoding='utf-8-sig')
    source=(e/'o7_source_sass.csv').read_text(encoding='utf-8-sig')
    result=analyze_capture(raw,source,receipt)
    assert result['static_fingerprint_verified'] and result['dynamic_instructions']==105521152
    for field,value in (('expected_kernel',SYMBOLS[1]),('numerical_checks_passed',False),
        ('packed_candidate_verified',False),('variant','o8')):
        bad=copy.deepcopy(receipt);bad[field]=value
        with pytest.raises(ValueError):analyze_capture(raw,source,bad)


def test_profile_is_fifty_warmups_one_capture_and_no_event_promotion():
    text=(ROOT/'scripts/run_register_layout_ncu.py').read_text()
    assert "'--launch-skip','50','--launch-count','1'" in text
    assert "'--cache-control','all','--clock-control','none'" in text
    assert 'new_performance_result=False' in text
    child=(ROOT/'scripts/profile_register_layout_kernel.py').read_text()
    assert "driver.run(case,2,'compute_only',0,1,2)" in child
    assert "driver.run(case,a.policy,'compute_only',50,1,2)" in child
    assert "torch.equal(out.view(torch.int32),ref.view(torch.int32))" in child
    assert "directory=ROOT/'reports/o378_roof_v85_codegen_checked'" in child

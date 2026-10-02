"""Offline wrapper contracts; GPU arithmetic is checked by the separate oracle."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_o78_fused_prepare", ROOT / "scripts/benchmark_o78_fused_prepare.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_prepare_and_timing_metadata_merge_without_duplicate_keywords(monkeypatch):
    monkeypatch.setattr(probe.OriginalDriver, "prepare", lambda self, case: {
        "ctas": 1, "integer_ctas": 1, "fallback_ctas": 0, "invalid_ctas": 0,
    })
    reference = np.array([[1, 2, 3]], dtype=np.uint32)
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: reference))
    case = SimpleNamespace(group_squares_reference={"asq": reference}, state={"asq": tensor})
    driver = probe.Driver.__new__(probe.Driver)
    guard = driver.prepare(case)
    for mode in probe.base.MODES:
        timing = probe.timing_contract(mode, 100)
        assert set(guard).isdisjoint(timing)
        record = dict(**guard, **timing)
        assert record["group_squares_exact"]
        assert record["preparation_implementation"] == "fused_conversion_group_squares_then_metadata"
        assert record["square_scratch_bytes_for_4096"] == 1048576
        assert not record["payload_reread_for_metadata"]


def test_import_does_not_replace_base_harness_classes():
    assert probe.base.Case is probe.OriginalCase
    assert probe.base.Driver is probe.OriginalDriver
    assert probe.base.timing_contract is probe.original_timing_contract


def test_fused_probe_is_not_in_formal_extension_build():
    for path in ("setup.py", "csrc/bindings.cpp", "csrc/sm80/o1_o3.cu"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "roof_o78_fused_prepare" not in source
        assert "roof_vector_norm_conversion" not in source

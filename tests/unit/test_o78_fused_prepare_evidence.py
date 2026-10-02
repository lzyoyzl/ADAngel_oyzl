"""Recompute unfiltered v69 evidence; this is not a substitute for GPU checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence/a100_o378_roof_v69"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))
SPEC = importlib.util.spec_from_file_location(
    "benchmark_o78_fullk_gpu_prepare", ROOT / "scripts/benchmark_o78_fullk_gpu_prepare.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
from benchmark_a100_o1 import stats


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("name,samples", [("screen", 4), ("trace24", 24)])
def test_every_raw_measurement_and_complete_summary(name, samples):
    run = EVIDENCE / "runs" / f"o378_roof_v69_{name}"
    if not run.exists():
        pytest.skip("v69 evidence has not been downloaded")
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert len(rows) == samples * 2 * 4 * 2
    assert len({r["sample_id"] for r in rows}) == samples
    for row in rows:
        assert row["metadata_exact"] and row["finite_fp32"] and row["MSE_regression_passed"]
        assert row["guard"]["invalid_ctas"] == 0
        assert row["preparation_implementation"] == "fused_conversion_group_squares_then_metadata"
        assert row["square_scratch_bytes_for_4096"] == 1048576
        assert not row["payload_reread_for_metadata"]
        assert row["round"] == 0
        for key, value in probe.timing_contract(row["mode"], 100).items():
            assert row[key] == value
        for stage, raw in row["raw_ms"].items():
            assert len(raw) == 200
            assert np.isfinite(raw).all() and min(raw) > 0
            expected = stats(raw)
            for key, value in expected.items():
                assert row["stage_summaries"][stage][key] == pytest.approx(value, rel=1e-12, abs=1e-12)
        assert row["summary"] == row["stage_summaries"][row["selected_stage"]]
        if row["mode"] == "conversion_only":
            assert np.allclose(row["raw_ms"]["total"],
                np.asarray(row["raw_ms"]["weight_conversion"]) + row["raw_ms"]["activation_conversion"],
                rtol=1e-6, atol=1e-9)
        if row["mode"] == "compute_only":
            assert row["raw_ms"]["total"] == row["raw_ms"]["gemm"]
        if row["candidate"] == 0:
            assert row["max_abs_vs_best"] == row["mse_vs_best"] == 0
        assert np.isclose(row["mse_vs_paired_fp16"], row["current_best_mse"], rtol=1e-5, atol=1e-12)
    assert probe.summarize(rows) == read(run / "summary.json")["records"]
    env = read(run / "environment.json")
    assert env["no_filtering"] and not env["production_default_changed"]
    assert env["extension_sha256"] == "94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462"
    assert env["args"]["warmup"] == 50 and env["args"]["repeats"] == 200 and env["args"]["inner"] == 100
    for entry in env["gemm_codegen"]["entries"].values():
        assert entry["native_u4_s4"] and entry["native_s4_s4"] and not entry["int8_mma"]


def test_preparation_receipt_and_gpu_validation_scope():
    reports = EVIDENCE / "reports"
    if not reports.exists():
        pytest.skip("v69 evidence has not been downloaded")
    build = reports / "o378_roof_v69_codegen"
    receipt = read(build / "build.json")
    for name, digest in receipt["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    # The binary lives in the downloaded archive, not in Git. Verify when present.
    library = build / "libo78_gpu_prepare.so"
    if library.exists():
        assert hashlib.sha256(library.read_bytes()).hexdigest() == receipt["driver_sha256"]
    for name in ("validation_fixed", "memcheck", "synccheck"):
        data = read(reports / f"o378_roof_v69_{name}" / "validation.json")
        assert data["passed"] and data["count"] == 64 and data["edge_count"] == 12
        assert all(r["group_squares_exact"] for r in data["checks"] + data["edge_checks"])
    for name in ("memcheck", "synccheck"):
        assert "ERROR SUMMARY: 0 errors" in (reports / f"o378_roof_v69_{name}.log").read_text()

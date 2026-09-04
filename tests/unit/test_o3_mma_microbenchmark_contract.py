from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_microbenchmark_keeps_all_three_mma_semantics() -> None:
    source = (ROOT / "benchmarks" / "o3_mma_lowering.cu").read_text()
    assert "SM80_16x8x64_S32U4S4S32_TN" in source
    assert "SM80_16x8x64_S32S4S4S32_TN" in source
    assert "SM80_16x8x32_S32S8S8S32_TN" in source
    assert "adangel_o3_mma_micro_u4s4_c1" in source
    assert "adangel_o3_mma_micro_u4s4_c4" in source
    assert "adangel_o3_mma_micro_s4s4_c1" in source
    assert "adangel_o3_mma_micro_s4s4_c4" in source
    assert "adangel_o3_mma_micro_s8s8_c1" in source
    assert "adangel_o3_mma_micro_s8s8_c4" in source


def test_microbenchmark_has_no_gemm_pipeline_or_scale_work() -> None:
    source = (ROOT / "benchmarks" / "o3_mma_lowering.cu").read_text()
    assert "PipelineTmaAsync" not in source
    assert "decode_ue8m0" not in source
    assert "a_scale" not in source
    assert "w_scale" not in source
    assert "cudaEventElapsedTime" in source


def test_microbenchmark_runner_audits_ptx_and_sass() -> None:
    runner = (ROOT / "scripts" / "run_o3_mma_lowering_microbenchmark.sh").read_text()
    assert "--dump-ptx" in runner
    assert "--dump-sass" in runner
    assert "--dump-resource-usage" in runner
    assert "ptx_u4s4" in runner
    assert "sass_u8s8_imma" in runner
    assert "logical_throughput_vs_s8_chain4" in runner

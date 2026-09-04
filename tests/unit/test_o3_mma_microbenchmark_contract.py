from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_microbenchmark_keeps_all_three_mma_semantics() -> None:
    source = (ROOT / "benchmarks" / "o3_mma_lowering.cu").read_text()
    assert "SM80_16x8x64_S32U4S4S32_TN" in source
    assert "SM80_16x8x64_S32S4S4S32_TN" in source
    assert "SM80_16x8x32_S32S8S8S32_TN" in source
    assert "SM80_8x8x32_S32U4S4S32_TN" in source
    assert "SM80_8x8x32_S32S4S4S32_TN" in source
    assert "SM80_8x8x16_S32S8S8S32_TN" in source
    assert "SM80_16x8x32_S32U4S4S32_TN" in source
    assert "SM80_16x8x32_S32S4S4S32_TN" in source
    assert "SM80_16x8x16_S32S8S8S32_TN" in source
    assert "adangel_o3_mma_micro_u4s4_c1" in source
    assert "adangel_o3_mma_micro_u4s4_c4" in source
    assert "adangel_o3_mma_micro_s4s4_c1" in source
    assert "adangel_o3_mma_micro_s4s4_c4" in source
    assert "adangel_o3_mma_micro_s8s8_c1" in source
    assert "adangel_o3_mma_micro_s8s8_c4" in source
    assert "adangel_o3_mma_micro_m8n8_u4s4_c4" in source
    assert "adangel_o3_mma_micro_m8n8_s4s4_c4" in source
    assert "adangel_o3_mma_micro_m8n8_s8s8_c4" in source
    assert "adangel_o3_mma_micro_m16n8k32_u4s4_c4" in source
    assert "adangel_o3_mma_micro_m16n8k32_s4s4_c4" in source
    assert "adangel_o3_mma_micro_m16n8k32_s8s8_c4" in source
    assert "SM80_16x8x64_S32S4U4S32_TN" in source
    assert "adangel_o3_mma_micro_split_pair_c1" in source
    assert "adangel_o3_mma_micro_split_pair_c4" in source
    assert "same signed-Q4 B fragment" in source


def test_microbenchmark_has_no_gemm_pipeline_or_scale_work() -> None:
    source = (ROOT / "benchmarks" / "o3_mma_lowering.cu").read_text()
    assert "PipelineTmaAsync" not in source
    assert "decode_ue8m0" not in source
    assert "a_scale" not in source
    assert "w_scale" not in source
    assert "cudaEventElapsedTime" in source
    assert "Different C fragments are essential" in source
    assert "0x01020408u * (chain + 1)" in source
    assert "2166136261u" in source
    assert "16777619u" in source


def test_microbenchmark_runner_audits_ptx_and_sass() -> None:
    runner = (ROOT / "scripts" / "run_o3_mma_lowering_microbenchmark.sh").read_text()
    assert "--dump-ptx" in runner
    assert "--dump-sass" in runner
    assert "--dump-resource-usage" in runner
    assert "ptx_u4s4" in runner
    assert "sass_u8s8_imma" in runner
    assert "logical_throughput_vs_s8_chain4" in runner
    assert "analyze_o3_mma_lowering.py" in runner


def test_static_attribution_script_counts_required_instructions() -> None:
    analyzer = (ROOT / "scripts" / "analyze_o3_mma_lowering.py").read_text()
    assert '"IMMA", "LOP3", "SHF", "IMAD"' in analyzer
    assert '"ptx_mma_static"' in analyzer
    assert '"outlined_lowering_per_ptx_mma"' in analyzer
    assert '"direct_imma_per_ptx_mma"' in analyzer
    assert 'instruction["mnemonic"] == "RET"' in analyzer


def test_toolchain_comparison_preserves_formal_o3_counting() -> None:
    comparison = (ROOT / "scripts" / "compare_o3_mma_toolchains.py").read_text()
    assert '"u4s4": "adangel_o3_mma_micro_u4s4_c4"' in comparison
    assert '"s4s4": "adangel_o3_mma_micro_s4s4_c4"' in comparison
    assert "2 * sum(u4_core.values()) + 2 * sum(s4_core.values())" in comparison
    assert "4096 // 128" in comparison
    assert '"floor_meets_target"' in comparison

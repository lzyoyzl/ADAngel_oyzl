from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_o3_int8x2_is_explicitly_diagnostic_and_keeps_production_unchanged():
    source = (ROOT / "csrc/sm120/o3_int8_split_diagnostic.cu").read_text(
        encoding="utf-8"
    )
    production = (ROOT / "csrc/sm120/o3_gemm.cu").read_text(encoding="utf-8")
    assert "SM80_16x8x32_S32S8S8S32_TN" in source
    assert "adangel_o3_split_int8x2_tma_ws" in source
    assert 'result["diagnostic_only"] = true' in source
    assert 'result["production_selected"] = false' in source
    assert 'result["o3_requirement_compliant"] = false' in source
    assert 'result["logical_int8_paths_per_group"] = 2' in source
    assert "tCrLowGroup(item) + 16 * tCrHighGroup(item)" in source
    assert (
        'kProductionO3Implementation = "m64_n32_k128_aligned_factor_16w"'
        in production
    )


def test_o3_int8x2_is_built_bound_validated_and_profiled():
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    bindings = (ROOT / "csrc/bindings.cpp").read_text(encoding="utf-8")
    validation = (ROOT / "scripts/validate_o3_int8x2.py").read_text(
        encoding="utf-8"
    )
    profiler = (ROOT / "scripts/profile_ncu_kernel.py").read_text(encoding="utf-8")
    assert '"csrc/sm120/o3_int8_split_diagnostic.cu"' in setup
    assert '"_benchmark_o3_split_int8x2"' in bindings
    assert "mse_vs_o3_semantic_reference" in validation
    assert '"o3_int8x2": "regex:adangel_o3_split_int8x2_tma_ws"' in profiler

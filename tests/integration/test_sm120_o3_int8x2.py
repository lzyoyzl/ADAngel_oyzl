import pytest


@pytest.mark.sm120
def test_o3_int8x2_diagnostic_matches_formal_o3_semantics():
    import torch

    from adangel.ops.extension import require_variant
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4

    generator = torch.Generator().manual_seed(2026)
    activation = torch.randn((128, 128), dtype=torch.float16, generator=generator)
    weight = torch.randn((64, 128), dtype=torch.float16, generator=generator)
    a_int8, a_scale = quantize_int8_per_row(activation)
    w_g128, w_scale = quantize_mxfp4(weight, group_size=128)
    native = require_variant("o3")
    args = (
        a_int8.cuda(),
        a_scale.cuda(),
        w_g128.cuda(),
        w_scale.cuda(),
    )
    formal = native._benchmark_o3_impl(
        "production", "compute_only", *args, 1, 2, 100
    )
    diagnostic = native._benchmark_o3_split_int8x2(
        "compute_only", *args, 1, 2, 100
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        diagnostic["output"], formal["output"], rtol=1e-3, atol=1e-3
    )
    kernel = dict(diagnostic["kernel"])
    assert kernel["mma_family"] == "IMMA_INT8_X2"
    assert tuple(kernel["cta_tile"]) == (128, 64, 128)
    assert kernel["logical_int8_paths_per_group"] == 2
    assert kernel["diagnostic_only"] is True
    assert kernel["production_selected"] is False
    assert kernel["o3_requirement_compliant"] is False

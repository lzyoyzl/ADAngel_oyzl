"""Exact CPU guard tests for independent O7/O8 cached full-K screening."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts/o78_fullk_integer_metadata.py"
_SPEC = importlib.util.spec_from_file_location("o78_fullk_integer_metadata", _PATH)
metadata = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metadata)


def prepare(a, w, aq, wq, **kwargs):
    return metadata.prepare_fullk_metadata(
        a, w, aq, wq, activation_kind=kwargs.pop("activation_kind", "ue8m0"),
        weight_kind=kwargs.pop("weight_kind", "e4m3"), **kwargs)


def test_all_scale_codes_exact_dyadics():
    for kind, maximum in (("ue8m0", 254), ("e4m3", 126), ("e6m2", 254)):
        for code in range(maximum + 1):
            mant, exp = metadata.dyadic_scale(np.uint8(code), kind)
            assert not mant or mant > 0 and mant % 2 == 1
            if kind == "ue8m0":
                expected = 2.0 ** (code - 127)
            elif kind == "e4m3":
                field, frac = code >> 3, code & 7
                expected = (8 + frac) * 2.0 ** (field - 10) if field else frac * 2.0**-9
            else:
                expected = (4 + (code & 3)) * 2.0 ** ((code >> 2) - 50)
            assert mant * 2.0**exp == expected


def test_safe_signed_terms_and_every_prefix():
    aq = np.array([[[2, -3], [-4, 1], [5, -2]], [[-1, 2], [3, -2], [-4, 1]]], dtype=np.int64)
    wq = np.array([[[3, -2], [2, -1], [-2, 4]], [[-2, 1], [-3, 2], [1, -1]]], dtype=np.int64)
    a = np.array([[127, 128, 129], [128, 127, 129]], dtype=np.uint8)
    w = np.array([[56, 60, 64], [60, 56, 64]], dtype=np.uint8)
    result = prepare(a, w, np.square(aq).sum(-1), np.square(wq).sum(-1), activation_base_multiplier=4, weight_base_multiplier=np.float32(0.25), tile_m=1, tile_n=1)
    assert np.all(result["status"] == 0)
    assert result["activation_factors"].shape == (3, 2)
    assert result["activation_factors"].dtype == np.int32
    assert result["activation_factors"].flags.c_contiguous
    for row in range(2):
        for col in range(2):
            acc, exact_real = 0, 0.0
            for g in range(3):
                p = sum(int(x) * int(y) for x, y in zip(aq[row, g], wq[col, g]))
                coef = int(result["activation_factors"][g, row]) * int(result["weight_factors"][g, col])
                assert coef <= metadata.INT32_MAX
                acc += p * coef
                assert abs(acc) <= metadata.INT32_MAX
                am, ae = metadata.dyadic_scale(a[row, g], "ue8m0")
                wm, we = metadata.dyadic_scale(w[col, g], "e4m3")
                exact_real += p * am * wm * 2.0 ** (ae + we)
            restored = acc * float(result["activation_bases"][row]) * float(result["weight_bases"][col])
            assert restored == exact_real


def test_anchor_includes_zero_payload_groups():
    side = metadata.operand_metadata([[127, 132]], [[0, 1]], "ue8m0")
    assert side["anchors"].tolist() == [0]
    assert side["logical_factors"] == [(1, 32)]
    assert side["norm2"] == [1024]


def test_zero_e4m3_and_zero_payload_but_ue8_code_zero_not_zero():
    side = metadata.operand_metadata([[0, 0]], [[128, 128]], "e4m3")
    assert side["norm2"] == [0] and not side["factors"].any()
    assert side["row_status"].tolist() == [0]
    assert metadata.dyadic_scale(0, "ue8m0") == (1, -127)
    assert prepare([[127, 127]], [[0, 0]], [[0, 0]], [[0, 0]])["status_flat"].tolist() == [0]


@pytest.mark.parametrize("kind,code", [("ue8m0", 255), ("e4m3", 127), ("e4m3", 128), ("e6m2", 255)])
def test_invalid_codes_precede_fallback(kind, code):
    result = prepare([[127], [0]], [[code]], [[0], [0]], [[0]], weight_kind=kind, tile_m=1, tile_n=1)
    assert result["status_flat"].tolist() == [2, 2]


def test_factor_overflow_exact_not_truncated():
    side = metadata.operand_metadata([[127, 159]], [[1, 1]], "ue8m0")
    assert side["logical_factors"] == [(1, 2**32)]
    assert side["norm2"] == [1 + 2**64]
    assert side["row_status"].tolist() == [1]
    assert not side["factors"].any()
    assert prepare([[127, 159]], [[56, 56]], [[1, 1]], [[1, 1]])["status_flat"].tolist() == [1]


def test_coefficient_guard_even_when_norm_zero():
    result = prepare([[127, 157]], [[1, 126]], [[0, 0]], [[0, 0]])
    assert result["activation"]["row_status"].tolist() == [0]
    assert result["weight"]["row_status"].tolist() == [0]
    assert result["cta_reasons"] == [["coefficient_exceeds_int32"]]


def test_prefix_guard_unbounded_host_ints_and_no_cancellation_assumption():
    result = prepare([[127, 127]], [[56, 56]], [[2**80, 2**80]], [[1, 1]])
    assert result["activation"]["norm2"] == [2**81]
    assert result["cta_reasons"] == [["absolute_sum_bound_exceeds_int32"]]
    # These two signed products cancel, but the first prefix would overflow.
    q = metadata.INT32_MAX
    result = prepare([[127]], [[56]], [[q*q + q*q]], [[2]])
    assert result["status_flat"].tolist() == [1]


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e40])
def test_multiplier_normal_positive_guard(multiplier):
    side = metadata.operand_metadata([[127]], [[0]], "ue8m0", multiplier)
    assert side["row_status"].tolist() == [1]
    assert np.isfinite(side["bases"]).all()


def test_anchor_base_overflow_subnormal_guard():
    assert metadata.operand_metadata([[0]], [[0]], "ue8m0")["row_status"].tolist() == [1]
    assert metadata.operand_metadata([[254]], [[0]], "ue8m0", 2.0)["row_status"].tolist() == [1]
    normal = metadata.operand_metadata([[1]], [[0]], "ue8m0")
    assert normal["row_status"].tolist() == [0]
    assert normal["bases"][0] == np.finfo(np.float32).tiny


def test_o8_common_multiplier_exact_reconstruction():
    result = prepare([[56, 60]], [[192, 194]], [[1, 1]], [[1, 1]], activation_kind="e4m3", weight_kind="e6m2", activation_base_multiplier=np.float32(8.0) / 4)
    assert result["status_flat"].tolist() == [0]
    for g in range(2):
        am, ae = metadata.dyadic_scale([56, 60][g], "e4m3")
        wm, we = metadata.dyadic_scale([192, 194][g], "e6m2")
        restored = int(result["activation_factors"][g, 0]) * int(result["weight_factors"][g, 0]) * float(result["activation_bases"][0]) * float(result["weight_bases"][0])
        assert restored == am * wm * 2.0 ** (ae + we) * 2


def test_y_major_mask_invalid_only_affects_its_row_tiles():
    result = prepare([[127], [255], [127]], [[56], [56], [56]], [[0], [0], [0]], [[0], [0], [0]], tile_m=1, tile_n=2)
    assert result["status"].tolist() == [[0, 0], [2, 2], [0, 0]]
    assert result["status_flat"].tolist() == [0, 0, 2, 2, 0, 0]


@pytest.mark.parametrize("squares", [[[1.0]], [[True]], [[-1]]])
def test_squares_exact_nonnegative_integers(squares):
    with pytest.raises(ValueError, match="nonnegative exact integers"):
        metadata.operand_metadata([[127]], squares, "ue8m0")


def test_shapes_tiles():
    with pytest.raises(ValueError, match="matching nonempty"):
        metadata.operand_metadata([], [], "ue8m0")
    with pytest.raises(ValueError, match="group counts"):
        prepare([[127, 127]], [[56]], [[0, 0]], [[0]])
    with pytest.raises(ValueError, match="positive integer CTA"):
        prepare([[127]], [[56]], [[0]], [[0]], tile_m=0)
    with pytest.raises(ValueError, match="shape"):
        metadata.operand_metadata([[127]], [[0]], "ue8m0", [1.0, 2.0])

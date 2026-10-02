"""Exact CPU guard for independent, cached O7/O8 full-K INT32 screening.

This is not an online CUDA guard or an end-to-end timing claim. Source scales
are factored in real arithmetic; the FP32 reassociation may change rounding.
O7 caller multipliers: activation=4, weight=T_W (common 4*T_W).
O8 caller multipliers: activation=T_A/4, weight=1 (common T_A/4).
Neither payload quantizer nor any G128 source scale is changed.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

INT32_MAX = 2**31 - 1
STATUS_INTEGER = np.uint32(0)
STATUS_FALLBACK = np.uint32(1)
STATUS_INVALID = np.uint32(2)
_FLOAT32 = np.finfo(np.float32)
_KINDS = {"ue8m0", "e4m3", "e6m2"}


def dyadic_scale(code: int, kind: str) -> tuple[int, int]:
    """Positive scale = odd mantissa * 2**exponent; zero is (0, 0).

    e6m2 means the project's HiF4 scale (bias 48, no zero/subnormals).
    """
    if kind not in _KINDS:
        raise ValueError(f"unsupported source-scale kind: {kind}")
    if isinstance(code, (bool, np.bool_)) or not isinstance(code, Integral):
        raise ValueError("integer scale code required (not float/bool)")
    code = int(code)
    if kind == "ue8m0" and 0 <= code <= 254:
        return 1, code - 127
    if kind == "e4m3" and 0 <= code <= 126:
        field, frac = code >> 3, code & 7
        mant, exp = (8 + frac, field - 10) if field else (frac, -9)
    elif kind == "e6m2" and 0 <= code <= 254:
        mant, exp = 4 + (code & 3), (code >> 2) - 50
    else:
        raise ValueError("invalid/nonpositive/NaN scale encoding")
    if mant == 0:
        return 0, 0
    while mant % 2 == 0:
        mant //= 2
        exp += 1
    return mant, exp


def _normal_positive(value: float) -> bool:
    return math.isfinite(value) and float(_FLOAT32.tiny) <= value <= float(_FLOAT32.max)


def _multipliers(value, rows: int) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim == 0:
        source = np.full(rows, source.item())
    if source.shape != (rows,):
        raise ValueError("base multiplier must be scalar or have shape [rows]")
    if source.dtype.kind not in "fiu":
        raise ValueError("real numeric base multiplier required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.asarray(source, dtype=np.float32)


def operand_metadata(scale_codes, group_sum_squares, kind: str, base_multiplier=1.0) -> dict:
    """Build exact factors/norms from matching source [rows, groups] arrays.

    Logical factors/norms are unbounded Python integers. Only proven-safe
    factors are copied to INT32; oversized values are never truncated. Anchor
    selection includes all nonzero scales, including zero-payload groups.
    Invalid codes reject a row; factor/base range failure selects fallback.
    """
    if kind not in _KINDS:
        raise ValueError(f"unsupported source-scale kind: {kind}")
    codes = np.asarray(scale_codes, dtype=object)
    squares = np.asarray(group_sum_squares, dtype=object)
    if codes.ndim != 2 or min(codes.shape, default=0) <= 0 or squares.shape != codes.shape:
        raise ValueError("matching nonempty [rows, groups] arrays required")
    rows, groups = codes.shape
    multipliers = _multipliers(base_multiplier, rows)
    factors = np.zeros((groups, rows), dtype=np.int32)
    bases = np.ones(rows, dtype=np.float32)
    anchors = np.zeros(rows, dtype=np.int32)
    status = np.zeros(rows, dtype=np.uint32)
    norms, maxima, logical_factors, reasons = [], [], [], []
    for row in range(rows):
        sums = squares[row].tolist()
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or int(v) < 0 for v in sums):
            raise ValueError("group sum of squares must contain nonnegative exact integers")
        try:
            values = [dyadic_scale(v, kind) for v in codes[row]]
        except ValueError:
            status[row] = STATUS_INVALID
            norms.append(0)
            maxima.append(0)
            logical_factors.append(tuple(0 for _ in range(groups)))
            reasons.append("invalid_source_scale")
            continue
        anchor = min((exp for mant, exp in values if mant), default=0)
        exact_factors = tuple(mant << (exp - anchor) if mant else 0 for mant, exp in values)
        norm = sum(int(square) * f * f for square, f in zip(sums, exact_factors))
        maximum = max(exact_factors)
        anchors[row] = anchor
        norms.append(norm)
        maxima.append(maximum)
        logical_factors.append(exact_factors)
        multiplier = float(multipliers[row])
        reason = None
        if maximum > INT32_MAX:
            reason = "factor_exceeds_int32"
        elif not _normal_positive(multiplier):
            reason = "base_multiplier_not_normal_positive_fp32"
        else:
            real_base = math.ldexp(multiplier, anchor)
            if not _normal_positive(real_base):
                reason = "anchor_base_not_normal_positive_fp32"
            else:
                # Adjusting a normal FP32 value by a power of two is exact
                # when the result stays normal; no mantissa is dropped.
                bases[row] = np.float32(real_base)
                factors[:, row] = exact_factors
        if reason is not None:
            status[row] = STATUS_FALLBACK
        reasons.append(reason)
    return dict(kind=kind, factors=np.ascontiguousarray(factors), bases=bases,
                anchors=anchors, norm2=norms, factor_max=maxima,
                logical_factors=logical_factors, row_status=status,
                row_reasons=reasons, source_shape=[rows, groups])


def prepare_fullk_metadata(
    activation_scale_codes, weight_scale_codes,
    activation_group_sum_squares, weight_group_sum_squares, *,
    activation_kind: str, weight_kind: str,
    activation_base_multiplier=1.0, weight_base_multiplier=1.0,
    tile_m: int = 64, tile_n: int = 128,
) -> dict:
    """Construct group-major factors and a CTA-uniform y-major guard mask.

    Status 0 proves every product and every INT32 prefix safe. Status 1 selects
    unchanged per-G128 FP32 fallback. Status 2 rejects invalid source codes.
    Failure of a sufficient bound is not evidence of an actual overflow.

    Cauchy bounds sum_g abs(P_g)*af_g*wf_g by sqrt(A_norm2*W_norm2).
    The independent coefficient guard protects INT32 af_g*wf_g. Neither
    proof uses FP square roots, capped shifts or fixed-width host sums.
    """
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0 for v in (tile_m, tile_n)):
        raise ValueError("positive integer CTA tiles required")
    a = operand_metadata(activation_scale_codes, activation_group_sum_squares, activation_kind, activation_base_multiplier)
    w = operand_metadata(weight_scale_codes, weight_group_sum_squares, weight_kind, weight_base_multiplier)
    m, groups = a["source_shape"]
    n, weight_groups = w["source_shape"]
    if groups != weight_groups:
        raise ValueError("activation and weight group counts must match")
    tile_m, tile_n = int(tile_m), int(tile_n)
    mask = np.zeros(((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n), dtype=np.uint32)
    reasons = []
    for ty, row_start in enumerate(range(0, m, tile_m)):
        tile_reasons = []
        a_slice = slice(row_start, min(row_start + tile_m, m))
        for tx, col_start in enumerate(range(0, n, tile_n)):
            w_slice = slice(col_start, min(col_start + tile_n, n))
            ast, wst = a["row_status"][a_slice], w["row_status"][w_slice]
            reason = None
            if np.any(ast == STATUS_INVALID) or np.any(wst == STATUS_INVALID):
                mask[ty, tx], reason = STATUS_INVALID, "invalid_source_scale"
            elif np.any(ast == STATUS_FALLBACK) or np.any(wst == STATUS_FALLBACK):
                mask[ty, tx], reason = STATUS_FALLBACK, "operand_factor_or_base_guard"
            elif max(a["factor_max"][a_slice]) * max(w["factor_max"][w_slice]) > INT32_MAX:
                mask[ty, tx], reason = STATUS_FALLBACK, "coefficient_exceeds_int32"
            elif max(a["norm2"][a_slice]) * max(w["norm2"][w_slice]) > INT32_MAX**2:
                mask[ty, tx], reason = STATUS_FALLBACK, "absolute_sum_bound_exceeds_int32"
            tile_reasons.append(reason)
        reasons.append(tile_reasons)
    return dict(activation=a, weight=w, activation_factors=a["factors"],
                weight_factors=w["factors"], activation_bases=a["bases"],
                weight_bases=w["bases"], status=np.ascontiguousarray(mask),
                status_flat=np.ascontiguousarray(mask.reshape(-1)),
                cta_reasons=reasons, cta_tile_mn=[tile_m, tile_n],
                shape=[m, n, groups],
                scope="cached_compute_only_CPU_oracle_not_online_preparation_timing")

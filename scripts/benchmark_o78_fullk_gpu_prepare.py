#!/usr/bin/env python3
"""v68 GPU-prepared full-K O7/O8 versus t59+conversion5, all four modes.

No defaults change. Source quantization and CPU oracle checks are outside
timing; GPU factor/norm/CTA guard work is INCLUDED in candidate conversion,
Cold and online activation work. Compute-only caches both operands/metadata.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("conversion_only", "compute_only", "cold", "steady_state")
SYMBOLS = ("adangel_roof_o78_fullk_control", "adangel_roof_o78_fullk_candidate")
STAGES = {
    "conversion_only": ("weight_conversion", "activation_conversion", "total"),
    "compute_only": ("gemm", "total"),
    "cold": ("weight_conversion", "activation_conversion", "gemm", "total"),
    "steady_state": ("activation_conversion", "gemm", "total"),
}
STATE_NAMES = ("a", "w", "as", "ws", "af", "wf", "ab", "wb", "an", "wn", "am", "wm", "ast", "wst", "status", "y")
SOURCE_NAMES = ("payload", "scale", "tensor_scale", "micro8", "micro4")
NORM_LIMIT = (2**31 - 1)**2 + 1


def timing_contract(mode, inner):
    if mode not in MODES or inner < 2:
        raise ValueError("supported mode and conversion inner >= 2 required")
    counts = {stage: inner if "conversion" in stage else 1 for stage in STAGES[mode]}
    counts["total"] = inner if mode == "conversion_only" else 1
    return dict(total_timing="sum_of_batched_stage_samples" if mode == "conversion_only" else "single_execution_cuda_event",
                stage_timing_inner_repeats=counts,
                weight_cached=mode in ("compute_only", "steady_state"),
                activation_prepared=mode == "compute_only",
                measurement_order="direct_path_then_isolated_conversions",
                timing_strategy="conversion_amortized_end_to_end_direct",
                timing_contract_version=2)


def normalize_timings(mode, raw, repeats):
    """Driver rows are W, A, GEMM, total. Reject malformed evidence, no filter."""
    values = np.asarray(raw, dtype=np.float64)
    if mode not in MODES or values.shape != (4, repeats) or repeats < 1:
        raise ValueError("four driver rows and exact repeat count required")
    indices = dict(weight_conversion=0, activation_conversion=1, gemm=2, total=3)
    result = {s: values[indices[s]].tolist() for s in STAGES[mode]}
    if any(not np.isfinite(v).all() or np.any(np.asarray(v) <= 0) for v in result.values()):
        raise ValueError("all selected timings must be finite and positive")
    if mode == "conversion_only" and not np.allclose(values[3], values[0] + values[1], rtol=1e-6, atol=1e-9):
        raise ValueError("conversion-only total must sum corresponding batch samples")
    if mode == "compute_only" and not np.array_equal(values[3], values[2]):
        raise ValueError("compute-only total must equal GEMM")
    return result


def expected_operand_gpu(cpu, activation=False):
    """GPU saturates large norms/maxima without falsely accepting an overflow."""
    result = dict(factors=cpu["factors"].copy(), bases=cpu["bases"].copy(),
                norms=np.asarray([NORM_LIMIT if int(maximum) > 2**31 - 1 else min(int(norm), NORM_LIMIT)
                                  for norm, maximum in zip(cpu["norm2"], cpu["factor_max"])], dtype=np.uint64),
                maxima=np.asarray([min(int(n), 2**31 - 1) for n in cpu["factor_max"]], dtype=np.int32),
                row_status=np.asarray(cpu["row_status"], dtype=np.uint32).copy())
    if activation:
        # INT32_MAX rounds up to 2^31 in FP32. A first epilogue multiply
        # by an activation base >=2^97 could overflow before W is applied.
        exponents = ((cpu["bases"].view(np.uint32) >> 23) & 255).astype(np.int32) - 127
        reject = (result["row_status"] == 0) & (exponents >= 97)
        result["row_status"][reject] = 1
        result["factors"][:, reject] = 0
        result["bases"][reject] = 1
    return result


def expected_v68_guard(cpu):
    """v68 adds a conservative FP32 epilogue range guard to the v67 proof."""
    expected_a = expected_operand_gpu(cpu["activation"], activation=True)
    expected_w = expected_operand_gpu(cpu["weight"])
    mask = cpu["status"].copy()
    tile_m, tile_n = cpu["cta_tile_mn"]
    ea = ((expected_a["bases"].view(np.uint32) >> 23) & 255).astype(np.int32) - 127
    ew = ((expected_w["bases"].view(np.uint32) >> 23) & 255).astype(np.int32) - 127
    for ty in range(mask.shape[0]):
        rows = slice(ty * tile_m, (ty + 1) * tile_m)
        for tx in range(mask.shape[1]):
            cols = slice(tx * tile_n, (tx + 1) * tile_n)
            if mask[ty, tx] != 0:
                continue
            if np.any(expected_a["row_status"][rows] != 0) or np.any(expected_w["row_status"][cols] != 0) or int(ea[rows].max()) + int(ew[cols].max()) > 94:
                mask[ty, tx] = 1
    return mask.reshape(-1)


def check_metadata_arrays(cpu, arrays):
    """Pure NumPy offline contract; compares every accepted/rejected row/tile."""
    for side, prefix in (("activation", "a"), ("weight", "w")):
        expected = expected_operand_gpu(cpu[side], activation=side == "activation")
        for key, state_key in (("factors", prefix + "f"), ("bases", prefix + "b"),
                               ("norms", prefix + "n"), ("maxima", prefix + "m"),
                               ("row_status", prefix + "st")):
            got = np.asarray(arrays[state_key])
            want = expected[key]
            if got.shape != want.shape or got.dtype != want.dtype:
                raise ValueError(f"GPU metadata shape/dtype mismatch: {state_key}")
            if key == "bases":
                equal = np.array_equal(got.view(np.uint32), want.view(np.uint32))
            else:
                equal = np.array_equal(got, want)
            if not equal:
                raise ValueError(f"GPU metadata oracle mismatch: {state_key}")
    got = np.asarray(arrays["status"], dtype=np.uint32)
    expected_guard = expected_v68_guard(cpu)
    if got.shape != expected_guard.shape or not np.array_equal(got, expected_guard):
        raise ValueError("GPU CTA guard differs from exact CPU proof")
    return dict(metadata_exact=True, integer_ctas=int(np.count_nonzero(got == 0)),
                fallback_ctas=int(np.count_nonzero(got == 1)), invalid_ctas=int(np.count_nonzero(got == 2)),
                ctas=int(got.size), preparation_scope="GPU_online_except_cached_weight_in_steady",
                additional_fp32_epilogue_range_guard=True,
                epilogue_guard_extra_fallback_ctas=int(np.count_nonzero((expected_guard == 1) & (cpu["status_flat"] == 0))))


def summarize(rows):
    if not rows:
        raise ValueError("empty evidence")
    index = {(r["sample_id"], r["variant"], r["mode"], r["round"], r["candidate"]): r for r in rows}
    ids = sorted({r["sample_id"] for r in rows})
    rounds = sorted({r["round"] for r in rows})
    expected = {(s, v, mode, ri, p) for s in ids for v in ("o7", "o8") for mode in MODES for ri in rounds for p in (0, 1)}
    if rounds != list(range(len(rounds))) or len(index) != len(rows) or set(index) != expected:
        raise ValueError("duplicate or incomplete paired four-mode evidence")
    from adangel.benchmark.metrics import bootstrap_median_ci
    output = []
    for variant in ("o7", "o8"):
        for mode in MODES:
            for policy in (0, 1):
                selected = [index[s, variant, mode, ri, policy] for s in ids for ri in rounds]
                if any(not r["metadata_exact"] or not r["MSE_regression_passed"] or not r["finite_fp32"] for r in selected):
                    raise ValueError("metadata or numerical acceptance failed")
                latencies = [statistics.median(index[s, variant, mode, ri, policy]["summary"]["median_ms"] for ri in rounds) for s in ids]
                speedups = [statistics.median(index[s, variant, mode, ri, 0]["summary"]["median_ms"] /
                           index[s, variant, mode, ri, policy]["summary"]["median_ms"] for ri in rounds) for s in ids]
                errors = [index[s, variant, mode, 0, policy]["mse_vs_paired_fp16"] for s in ids]
                output.append(dict(variant=variant, mode=mode, candidate=policy, samples=len(ids), records=len(selected),
                    median_ms=statistics.median(latencies), paired_speedup=statistics.median(speedups),
                    paired_speedup_ci95=list(bootstrap_median_ci(speedups, 10000, .95, 20261002)) if len(ids) > 1 else None,
                    median_mse=statistics.median(errors), mean_mse=statistics.mean(errors),
                    selected_cv_failed_records=sum(r["summary"]["cv_percent"] >= 3 for r in selected),
                    any_stage_cv_failed_records=sum(any(v["cv_percent"] >= 3 for v in r["stage_summaries"].values()) for r in selected),
                    max_abs_vs_best=max(r["max_abs_vs_best"] for r in selected),
                    max_mse_vs_best=max(r["mse_vs_best"] for r in selected),
                    fallback_ctas=max(r["guard"]["fallback_ctas"] for r in selected)))
    return output


def checked_gpu_build(directory):
    """Bind new GPU preparation library to its locally generated build receipt."""
    from adangel.trace.storage import sha256_file
    receipt = json.loads((directory / "build.json").read_text())
    lib = directory / "libo78_gpu_prepare.so"
    if sha256_file(lib) != receipt["driver_sha256"]:
        raise ValueError("GPU preparation library drift")
    for path, digest in receipt["sources"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"GPU preparation source drift: {path}")
    return lib, receipt


class Driver:
    def __init__(self, library, gemm_directory):
        from benchmark_o78_fullk_integer_probe import checked
        self.codegen = checked(gemm_directory)
        self.lib = ct.CDLL(str(library.resolve()))
        self.lib.roof_probe_error.restype = ct.c_char_p
        self.lib.roof_probe_open.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_uint, ct.POINTER(ct.c_void_p)]
        self.lib.roof_probe_close.argtypes = [ct.c_void_p]
        self.lib.roof_probe_resources.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        pointers = [ct.POINTER(ct.c_uint64)] * 3
        self.lib.roof_o78_gpu_prepare.argtypes = [ct.c_int, *pointers, ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_void_p]
        self.lib.roof_o78_gpu_benchmark.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, *pointers,
            ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.POINTER(ct.c_float)]
        self.handles, self.resources = {}, {}
        try:
            for policy, symbol in enumerate(SYMBOLS):
                handle = ct.c_void_p()
                self.check(self.lib.roof_probe_open(str((gemm_directory / "o78_fullk.cubin").resolve()).encode(), symbol.encode(), 34304, ct.byref(handle)))
                self.handles[policy] = handle
                values = (ct.c_int * 4)()
                self.check(self.lib.roof_probe_resources(handle, values))
                self.resources[policy] = dict(registers_per_thread=values[0], local_size_bytes=values[1], threads=values[2],
                    active_blocks_per_sm=values[3], shared_memory_bytes=34304, cta_tile=[64, 128, 128], pipeline_stages=2)
        except Exception:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.roof_probe_error().decode())

    def close(self):
        for handle in self.handles.values():
            self.check(self.lib.roof_probe_close(handle))
        self.handles.clear()

    def prepare(self, case):
        import torch
        self.check(self.lib.roof_o78_gpu_prepare(int(case.variant[1:]), case.a_source, case.w_source, case.state_pointers,
            case.m, case.n, case.a_multiplier, case.w_multiplier, torch.cuda.current_stream().cuda_stream))
        torch.cuda.current_stream().synchronize()
        # Bind metadata proof to the exact converted payload used by GEMM.
        if hasattr(case, "expected_payload"):
            for key, reference in case.expected_payload.items():
                assert torch.equal(case.state[key].view(torch.uint8), reference.view(torch.uint8)), key
        arrays = {key: case.state[key].cpu().numpy() for key in ("af", "wf", "ab", "wb", "an", "wn", "am", "wm", "ast", "wst", "status")}
        return check_metadata_arrays(case.oracle, arrays)

    def run(self, case, policy, mode, warmup, repeats, inner):
        import torch
        if policy not in (0, 1) or mode not in MODES:
            raise ValueError("supported independent policy/mode required")
        if np.any(case.oracle["status_flat"] == 2):
            raise ValueError("invalid source must not expose an unwritten output")
        values = (ct.c_float * (4 * repeats))()
        self.check(self.lib.roof_o78_gpu_benchmark(self.handles[policy], int(case.variant[1:]), policy, MODES.index(mode),
            case.a_source, case.w_source, case.state_pointers, case.m, case.n, case.a_multiplier, case.w_multiplier, warmup, repeats, inner,
            torch.cuda.current_stream().cuda_stream, values))
        raw = np.ctypeslib.as_array(values).reshape(4, repeats)
        return case.state["y"], normalize_timings(mode, raw, repeats)


class Case:
    """All device buffers and oracle are constructed before measured intervals."""
    def __init__(self, variant, weight_source, activation_source, base):
        import torch
        from adangel.quantization import mixed_formats as mf
        from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
        from validate_a100_split_grouped import pack_q4
        from o78_fullk_integer_metadata import prepare_fullk_metadata
        self.variant, self.wsrc, self.asrc = variant, weight_source, activation_source
        aq, asc = mf.to_fixed_reference(activation_source)
        wq, wsc = mf.to_fixed_reference(weight_source)
        assert torch.equal(base["converted_activation"][0], split_int8_to_packed_int4(aq))
        assert torch.equal(base["converted_weight"][0], pack_q4(wq))
        assert torch.equal(base["converted_activation"][1].view(torch.int32), asc.view(torch.int32))
        assert torch.equal(base["converted_weight"][1].view(torch.int32), wsc.view(torch.int32))
        self.m, k = activation_source["shape"]
        self.n, wk = weight_source["shape"]
        if k != 4096 or wk != 4096 or self.m % 64 or self.n % 128:
            raise ValueError("aligned M64/N128 and full K4096 required")
        ak, wk = ("ue8m0", "e4m3") if variant == "o7" else ("e4m3", "e6m2")
        tensor_scale = float((weight_source if variant == "o7" else activation_source)["tensor_scale"].item())
        self.a_multiplier = np.float32(4 if variant == "o7" else np.float32(tensor_scale) * np.float32(.25))
        self.w_multiplier = np.float32(tensor_scale if variant == "o7" else 1)
        squares = lambda q: q.reshape(q.shape[0], 32, 128).long().square().sum(-1).cpu().numpy()
        self.a_sumsq, self.w_sumsq = squares(aq), squares(wq)
        self.oracle = prepare_fullk_metadata(activation_source["scale"].cpu().numpy(), weight_source["scale"].cpu().numpy(),
            self.a_sumsq, self.w_sumsq, activation_kind=ak, weight_kind=wk,
            activation_base_multiplier=self.a_multiplier, weight_base_multiplier=self.w_multiplier)
        self.state = {}
        for key, shape, dtype in (
            ("a", (2, 32, self.m, 64), torch.uint8), ("w", (32, self.n, 64), torch.uint8),
            ("as", (32, self.m), torch.float32), ("ws", (32, self.n), torch.float32),
            ("af", (32, self.m), torch.int32), ("wf", (32, self.n), torch.int32),
            ("ab", (self.m,), torch.float32), ("wb", (self.n,), torch.float32),
            ("an", (self.m,), torch.uint64), ("wn", (self.n,), torch.uint64),
            ("am", (self.m,), torch.int32), ("wm", (self.n,), torch.int32),
            ("ast", (self.m,), torch.uint32), ("wst", (self.n,), torch.uint32),
            ("status", (self.oracle["status_flat"].size,), torch.uint32), ("y", (self.m, self.n), torch.float32)):
            self.state[key] = torch.empty(shape, dtype=dtype, device=aq.device)
        # CUDA preparation overwrites these values in its normal flow; copies
        # are for safe initial storage and diagnostic comparisons only.
        self.state["a"].copy_(base["packed_activation_g128_major"])
        self.state["w"].copy_(base["packed_weight_g128_major"])
        self.state["as"].copy_(asc.T)
        self.state["ws"].copy_(wsc.T)
        self.expected_payload = {key: self.state[key].clone() for key in ("a", "w", "as", "ws")}
        source_pointers = lambda source: (ct.c_uint64 * 5)(*(source[key].data_ptr() if key in source else 0 for key in SOURCE_NAMES))
        self.a_source = source_pointers(activation_source)
        self.w_source = source_pointers(weight_source)
        self.state_pointers = (ct.c_uint64 * 16)(*(self.state[key].data_ptr() for key in STATE_NAMES))

    def refresh_oracle(self):
        from o78_fullk_integer_metadata import prepare_fullk_metadata
        ak, wk = ("ue8m0", "e4m3") if self.variant == "o7" else ("e4m3", "e6m2")
        self.oracle = prepare_fullk_metadata(self.asrc["scale"].cpu().numpy(), self.wsrc["scale"].cpu().numpy(),
            self.a_sumsq, self.w_sumsq, activation_kind=ak, weight_kind=wk,
            activation_base_multiplier=self.a_multiplier, weight_base_multiplier=self.w_multiplier)

    def check_payload(self, base):
        import torch
        for key, want in (("a", base["packed_activation_g128_major"]), ("w", base["packed_weight_g128_major"]),
                          ("as", base["converted_activation"][1].T), ("ws", base["converted_weight"][1].T)):
            got = self.state[key]
            if got.dtype == torch.float32:
                assert torch.equal(got.view(torch.int32), want.view(torch.int32)), f"new conversion payload/scale mismatch: {key}"
            else:
                assert torch.equal(got, want), f"new conversion payload mismatch: {key}"


def validate_edges(driver):
    """GPU branches for invalid encodings, exact saturation and fallback.

    Source scales are deliberately mutated AFTER constructing a valid case;
    invalid sources are only prepared, never executed as a GEMM.
    """
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from o78_fullk_integer_metadata import prepare_fullk_metadata
    cases = []
    specs = [
        ("ue_code_zero", "o7"), ("factor_delta30_norm_uint64_overflow", "o7"),
        ("capped_norm_other_operand_zero", "o7"), ("factor_delta32", "o7"),
        ("invalid_ue255_and_overflow", "o7"), ("invalid_e4_127", "o7"),
        ("subnormal_multiplier", "o7"), ("activation_epilogue_overflow", "o7"),
        ("combined_epilogue_overflow", "o7"), ("e4_zero_scale", "o8"),
        ("e6_code_zero", "o8"), ("invalid_e6_255", "o8"),
    ]
    for name, variant in specs:
        wf, af = mf.VARIANTS[variant]
        a = torch.full((64, 4096), .5, device="cuda", dtype=torch.float16)
        w = torch.full((128, 4096), .125, device="cuda", dtype=torch.float16)
        if name == "capped_norm_other_operand_zero":
            w.zero_()
        ws, acs = mf.quantize_source(w, wf), mf.quantize_source(a, af)
        base = native._benchmark_mixed(variant, "compute_only", ws, acs, 0, 1, 2, "64x128x256", "group_major", 59, 5)
        case = Case(variant, ws, acs, base)
        aq, _ = mf.to_fixed_reference(acs); wq, _ = mf.to_fixed_reference(ws)
        acs["scale"].fill_(127 if variant == "o7" else 56)
        ws["scale"].fill_(56 if variant == "o7" else 192)
        case.a_multiplier = np.float32(4 if variant == "o7" else 1)
        case.w_multiplier = np.float32(1)
        if name == "ue_code_zero": acs["scale"].zero_()
        elif name in ("factor_delta30_norm_uint64_overflow", "capped_norm_other_operand_zero"):
            acs["scale"][:, -1] = 157
        elif name == "factor_delta32": acs["scale"][:, -1] = 159
        elif name == "invalid_ue255_and_overflow":
            acs["scale"][:, -1] = 159; acs["scale"][0, 0] = 255
        elif name == "invalid_e4_127": ws["scale"][0, 0] = 127
        elif name == "subnormal_multiplier": case.w_multiplier = np.float32(2.0**-149)
        elif name == "activation_epilogue_overflow": case.a_multiplier = np.float32(2.0**97)
        elif name == "combined_epilogue_overflow":
            case.a_multiplier = np.float32(2.0**48); case.w_multiplier = np.float32(2.0**47)
        elif name == "e4_zero_scale": acs["scale"].zero_()
        elif name == "e6_code_zero": ws["scale"].zero_()
        elif name == "invalid_e6_255": ws["scale"][0, 0] = 255
        squares = lambda q: q.reshape(q.shape[0], 32, 128).long().square().sum(-1).cpu().numpy()
        case.oracle = prepare_fullk_metadata(acs["scale"].cpu().numpy(), ws["scale"].cpu().numpy(),
            squares(aq), squares(wq), activation_kind="ue8m0" if variant == "o7" else "e4m3",
            weight_kind="e4m3" if variant == "o7" else "e6m2",
            activation_base_multiplier=case.a_multiplier, weight_base_multiplier=case.w_multiplier)
        # Only codes/base multipliers changed; payload itself must stay exact.
        case.expected_payload = {key: case.expected_payload[key] for key in ("a", "w")}
        guard = driver.prepare(case)
        if name == "factor_delta30_norm_uint64_overflow":
            assert max(case.oracle["activation"]["norm2"]) > 2**64
            assert guard["fallback_ctas"] == guard["ctas"]
        if name == "capped_norm_other_operand_zero":
            assert max(case.oracle["activation"]["norm2"]) > NORM_LIMIT
            assert guard["integer_ctas"] == guard["ctas"]
        if name.startswith("invalid_"):
            assert guard["invalid_ctas"] > 0
            try:
                driver.run(case, 1, "compute_only", 0, 1, 2)
            except ValueError as exc:
                assert "invalid source" in str(exc)
            else:
                raise AssertionError("invalid source admitted to GEMM")
        cases.append(dict(name=name, variant=variant, **guard))
    return cases


def validate(driver):
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from roof_reduction_validation import reference_fp64
    checks = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for pattern in ("random", "zero", "alternating", "wide_scale"):
            m, n, k = (128, 256, 4096) if pattern == "wide_scale" else (64, 128, 4096)
            torch.manual_seed(20261002)
            activation = (torch.randn(m, k, device="cuda") * .4).half()
            weight = (torch.randn(n, k, device="cuda") * .1).half()
            if pattern == "zero":
                activation.zero_(); weight.zero_()
            if pattern == "alternating":
                activation[:, ::2] = -8; activation[:, 1::2] = 7
                weight[:, ::2] = -7; weight[:, 1::2] = 6
            for variant, (wf, af) in mf.VARIANTS.items():
                ws, acs = mf.quantize_source(weight, wf), mf.quantize_source(activation, af)
                if pattern == "wide_scale":
                    if variant == "o7":
                        acs["scale"].fill_(127); acs["scale"][:, -1] = 159
                    else:
                        ws["scale"].fill_(1); ws["scale"][:, -1] = 192
                base = native._benchmark_mixed(variant, "compute_only", ws, acs, 0, 1, 2, "64x128x256", "group_major", 59, 5)
                case = Case(variant, ws, acs, base)
                guard = driver.prepare(case)
                case.check_payload(base)
                semantic = reference_fp64(variant, (*base["converted_activation"], *base["converted_weight"]))
                for policy in (0, 1):
                    for mode in MODES:
                        y, times = driver.run(case, policy, mode, 0, 2, 2)
                        if mode == "conversion_only":
                            # A representative independent GEMM gives a defined
                            # correctness output; never counted as conversion.
                            y, _ = driver.run(case, policy, "compute_only", 0, 1, 2)
                        torch.testing.assert_close(y.double(), semantic, rtol=1e-3, atol=1e-3)
                        if policy == 0 or guard["integer_ctas"] == 0:
                            assert torch.equal(y.view(torch.int32), base["output"].view(torch.int32))
                        assert y.dtype == torch.float32 and bool(torch.isfinite(y).all())
                        checks.append(dict(pattern=pattern, variant=variant, candidate=policy, mode=mode,
                            shape=[m, n, k], semantic_tolerance_passed=True, nondefault_stream=True,
                            **guard, **timing_contract(mode, 2)))
        stream.synchronize()
    edges = validate_edges(driver)
    return dict(passed=True, count=len(checks), checks=checks, edge_count=len(edges), edge_checks=edges,
        scope="GPU_preparation_and_four_modes_small_MN_full_K4096")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-build", type=Path, required=True)
    p.add_argument("--gemm-cubins", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, choices=(4, 24), default=4)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--inner", type=int, default=100)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--data", type=Path, default=Path("data/prepared/llama2_7b_prefill_o0_o4"))
    p.add_argument("--raw-data", type=Path, default=Path("data/raw/llama2_7b_prefill"))
    p.add_argument("--trace-config", type=Path, default=Path("configs/trace/llama2_7b_prefill.yaml"))
    args = p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT) or min(args.rounds, args.repeats) < 1 or args.inner < 2 or args.warmup < 0:
        p.error("fresh repository output and valid measurement counts required")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization import mixed_formats as mf
    from adangel.trace.storage import sha256_file, load_prepared
    from adangel.trace.prepare import _load_and_validate_raw
    from benchmark_a100_o1 import command, stats
    from benchmark_a100_roof_trace import measurement_order
    from benchmark_a100_mixed import validate_fp16_result
    from benchmark_a100_mixed_trace import inspect_inputs, inspect_raw_inputs, verify_raw_prepared, source_identity, mse
    from roof_reduction_validation import mse_regression_ok
    torch.cuda.init(); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability() == (8, 0)
    context_anchor = torch.empty(1, device="cuda")
    library, build = checked_gpu_build(args.gpu_build)
    args.output.mkdir(parents=True)
    def save(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    def append(name, obj):
        with (args.output / name).open("a") as f:
            f.write(json.dumps(obj, allow_nan=False) + "\n")
    driver = Driver(library, args.gemm_cubins)
    try:
        save("validation.json", validate(driver))
        if args.validate_only:
            return
        manifest, mh = inspect_inputs(args.data)
        raw, rh = inspect_raw_inputs(args.raw_data, manifest, args.trace_config)
        raw_index = {entry["sample_id"]: entry for entry in raw["samples"]}
        save("environment.json", dict(git_commit=command("git", "rev-parse", "HEAD"),
            extension_sha256=sha256_file(Path(native.__file__)), gemm_codegen=driver.codegen, gpu_preparation_build=build,
            resources=driver.resources, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            prepared_manifest_sha256=mh, raw_manifest_sha256=rh, production_default_changed=False,
            input_policy="original_FP16_direct_source_quantization_excluded_from_conversion",
            candidate="independent_GPU_guarded_full_K_INT32_with_exact_per_G128_source_scales",
            control="t59_GEMM_plus_conversion5", no_filtering=True, policy="unlocked_shared_GPU_cyclic_AB",
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
        rows = []
        for si, entry in enumerate(manifest["samples"][:args.samples]):
            re = raw_index[entry["sample_id"]]
            path, rp = args.data / entry["file"], args.raw_data / re["file"]
            assert sha256_file(path) == entry["sha256"] and sha256_file(rp) == re["sha256"]
            record = _load_and_validate_raw(rp, re["layer"], re["projection"])
            x = load_prepared(path, device="cpu")
            verify_raw_prepared(x, (record["activation_fp16"], record["weight_fp16"])); del x
            for vi, (variant, (wf, af)) in enumerate(mf.VARIANTS.items()):
                wsrc = mf.quantize_source(record["weight_fp16"].cuda(), wf)
                asrc = mf.quantize_source(record["activation_fp16"].cuda(), af)
                append("source_provenance.jsonl", dict(sample_id=entry["sample_id"], variant=variant,
                    raw_sha256=re["sha256"], weight=source_identity(wsrc), activation=source_identity(asrc)))
                paired = native._benchmark_mixed(mf.PAIRED_BASELINE[variant], "compute_only", wsrc, asrc, 0, 1, 2, "64x128x256", "row_major")
                validate_fp16_result(paired, wsrc, asrc)
                base = native._benchmark_mixed(variant, "compute_only", wsrc, asrc, 0, 1, 2, "64x128x256", "group_major", 59, 5)
                case = Case(variant, wsrc, asrc, base)
                guard = driver.prepare(case)
                case.check_payload(base)
                assert guard["invalid_ctas"] == 0
                expected_mse = mse(base["output"], paired["output"])
                for ri in range(args.rounds):
                    append("gpu_snapshots.jsonl", dict(sample_id=entry["sample_id"], variant=variant, round=ri, time=time.time(),
                        gpu=command("nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu", "--format=csv")))
                    for mi, mode in enumerate(MODES):
                        order = measurement_order((0, 1), si, vi + mi, ri)
                        for policy in order:
                            output, timings = driver.run(case, policy, mode, args.warmup, args.repeats, args.inner)
                            if mode == "conversion_only":
                                output, _ = driver.run(case, policy, "compute_only", 0, 1, 2)
                            assert output.dtype == torch.float32 and bool(torch.isfinite(output).all())
                            torch.testing.assert_close(output, base["output"], rtol=1e-3, atol=1e-3)
                            if policy == 0:
                                assert torch.equal(output.view(torch.int32), base["output"].view(torch.int32))
                            error = mse(output, paired["output"])
                            assert mse_regression_ok(error, expected_mse)
                            stage_summaries = {name: stats(values) for name, values in timings.items()}
                            selected = "gemm" if mode == "compute_only" else "total"
                            # Recheck actual GPU-generated metadata after timed
                            # work, without replacing it with CPU-oracle buffers.
                            arrays = {key: case.state[key].cpu().numpy() for key in ("af", "wf", "ab", "wb", "an", "wn", "am", "wm", "ast", "wst", "status")}
                            current_guard = check_metadata_arrays(case.oracle, arrays)
                            row = dict(sample_id=entry["sample_id"], variant=variant, mode=mode, round=ri, candidate=policy,
                                execution_order=list(order), raw_ms=timings, stage_summaries=stage_summaries,
                                summary=stage_summaries[selected], selected_stage=selected, metadata_exact=True, guard=current_guard,
                                finite_fp32=True, MSE_regression_passed=True, mse_vs_paired_fp16=error, current_best_mse=expected_mse,
                                paired_fp16=mf.PAIRED_BASELINE[variant], mse_vs_best=mse(output, base["output"]),
                                max_abs_vs_best=(output - base["output"]).abs().max().item(),
                                resources=driver.resources[policy], **timing_contract(mode, args.inner))
                            rows.append(row); append("results.jsonl", row)
                        print(entry["sample_id"], variant, mode, "paired complete", flush=True)
                del case, base, paired, wsrc, asrc, output
            assert len(rows) == (si + 1) * 2 * len(MODES) * args.rounds * 2
            save("summary.json", dict(scope="GPU_online_preparation_included_four_mode_paired_evidence", production_default_changed=False,
                no_filtering=True, correctness_passed=True, records=summarize(rows)))
        assert sha256_file(args.data / "manifest.json") == mh and sha256_file(args.raw_data / "trace_manifest.json") == rh
        print("GPU PREPARATION FOUR-MODE PAIRED TEST PASSED", flush=True)
    finally:
        driver.close()


if __name__ == "__main__":
    main()

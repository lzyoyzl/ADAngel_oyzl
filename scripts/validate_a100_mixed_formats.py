#!/usr/bin/env python3
"""Synthetic codec -> native fixed conversion -> dual-INT4 correctness acceptance.

Does not read/rewrite the user's prepared trace, publish formal O7/O8 MSE, or
time Python quantization as a native conversion. All output uses a fresh folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output directory")
    import torch
    from adangel import _sm80 as native
    if getattr(native, "mixed_experiment_naming_version", None) != 3:
        raise RuntimeError("rebuild SM80 extension for renamed O7/O8 and FP16 O5/O6")
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from validate_a100_split_grouped import reference
    from benchmark_a100_mixed import validate_fp16_result
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires A100 SM80")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(repo / "tests/unit"), "test_mixed_formats.py")
    with (args.output / "codec_tests.txt").open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise AssertionError("codec tests failed or skipped; inspect codec_tests.txt")
    conversion_checks = []
    fp16_checks = []
    bitplane_checks = []

    def move(source):
        return {key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in source.items()}

    def convert(source, label):
        decoded = mf.dequantize_source(source).half()
        if torch.isfinite(decoded).all():
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                half = native._dequantize_mixed_source(move(source))
            stream.synchronize()
            assert torch.equal(half.cpu().view(torch.int16), decoded.view(torch.int16)), (label, source["format"])
            fp16_checks.append({"case": label, "format": source["format"], "fp16_bitwise": True})
        else:
            try:
                native._dequantize_mixed_source(move(source))
            except RuntimeError as exc:
                assert "overflows FP16" in str(exc)
            else:
                raise AssertionError("nonfinite FP16 source silently accepted")
            fp16_checks.append({"case": label, "format": source["format"], "overflow_rejected": True})
        q, scale = mf.to_fixed_reference(source)
        weight = mf.FORMATS[source["format"]][1] == 4
        packed = mf._pack_nibbles(q.to(torch.uint8) & 15) if weight else split_int8_to_packed_int4(q)
        for layout in ("row_major", "group_major"):
            # Transfer, conversion and consumers use one non-default stream.
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                converted = native._convert_mixed_source(move(source), layout)
                planes = native._convert_mixed_bitplanes(move(source), layout)
            stream.synchronize()
            torch.testing.assert_close(converted["packed"].cpu(), packed, rtol=0, atol=0)
            assert torch.equal(converted["scale"].cpu().contiguous().view(torch.int32), scale.view(torch.int32))
            expected_stride = (scale.shape[1], 1) if layout == "row_major" else (1, scale.shape[0])
            assert converted["scale"].stride() == expected_stride
            bits = 4 if weight else (8 if source["format"] == "mxfp8_e4m3_g128" else 6)
            expected_planes = torch.stack([((((q.int().reshape(q.shape[0], -1, 32) >> b)&1).long()
                    << torch.arange(32)).sum(-1)).int() for b in range(bits)])
            assert planes["planes"] == bits
            assert torch.equal(planes["packed"].cpu(), expected_planes), (label, source["format"])
            assert torch.equal(planes["scale"].cpu().contiguous().view(torch.int32), scale.view(torch.int32))
            bitplane_checks.append({"case": label, "format": source["format"], "scale_layout": layout,
                                   "packed_bitwise": True, "scale_bitwise": True, "planes": bits})
            conversion_checks.append({"case": label, "format": source["format"], "scale_layout": layout,
                                      "shape": source["shape"], "packed_bitwise_equal": True,
                                      "scale_bitwise_equal": True})
            if layout == "row_major":
                row_converted = converted
        return row_converted, q, scale

    # Exhaust every finite payload code, including negative zero. For HiF4 cross
    # every payload with all four micro8/micro4 combinations, plus mixed layouts.
    for fmt, (kind, _, _) in mf.FORMATS.items():
        sign = mf.SIGN_BITS[kind]
        codes = [c for c in range(2 * sign) if not (kind == "e4m3" and c & 127 == 127)]
        values = torch.tensor((codes * 32)[:512], dtype=torch.uint8).reshape(2, 256)
        for pattern in range(4 if fmt == "hif4_g128" else 1):
            source = mf.quantize_source(torch.ones((2, 256)), fmt)
            source["payload"] = mf._pack_nibbles(values) if kind in ("e2m1", "s1p2") else values
            if fmt == "hif4_g128":
                source["micro8"].fill_(255 if pattern & 1 else 0)
                source["micro4"].fill_(255 if pattern & 2 else 0)
                source["scale"] = torch.tensor([[0, 180], [192, 254]], dtype=torch.uint8)
            elif fmt == "mxfp8_e4m3_g128":
                source["scale"] = torch.tensor([[0, 120], [127, 252]], dtype=torch.uint8)
            else:
                source["scale"] = torch.tensor([[0, 1], [64, 126]], dtype=torch.uint8)
                source["tensor_scale"] = torch.tensor([.0317], dtype=torch.float32)
            convert(source, f"all_payloads_micro_{pattern}")
        for rows in (1, 3):
            # Partial conversion CTA: guard threads beyond the final 64 pairs.
            x = torch.linspace(-8, 8, rows * 128).reshape(rows, 128)
            convert(mf.quantize_source(x, fmt), f"tail_rows_{rows}")

    gemm_checks = []
    timing_checks = []
    layout_checks = []
    binary_checks = []
    # O0's strict no-SIMT fallback policy has no eligible HMMA heuristic for
    # small M/N at K4096 on this A100; 512x512 preserves the large-K test and O0.
    for m, n, k in ((64, 128, 256), (128, 128, 512), (512, 512, 4096)):
        for pattern in ("random", "zero", "alternating"):
            gen = torch.Generator().manual_seed(5600 + k)
            a = (torch.randn((m, k), generator=gen) * .5).half()
            w = (torch.randn((n, k), generator=gen) * .125).half()
            if pattern == "zero":
                a.zero_(); w.zero_()
            elif pattern == "alternating":
                a[:, ::2] = -7.5; a[:, 1::2] = 7.5
                w[:, ::2] = -7; w[:, 1::2] = 7
            # Synthetic O0 from the SAME FP16 tensors, not from any real trace.
            ai, asc = quantize_int8_per_row(a)
            wm, wms = quantize_mxfp4(w)
            o0_run = native.benchmark_o0(ai.cuda(), asc.cuda(), wm.cuda(), wms.cuda(),
                                         "compute_only", 0, 1, 100)
            o0 = o0_run["output"].cpu()
            for variant, (wfmt, afmt) in mf.VARIANTS.items():
                wsrc, asrc = mf.quantize_source(w, wfmt), mf.quantize_source(a, afmt)
                fp16_case = mf.PAIRED_BASELINE[variant]
                gpu_w, gpu_a = move(wsrc), move(asrc)
                fp16_run = native._benchmark_mixed(fp16_case, "compute_only", gpu_w, gpu_a, 0, 2, 10)
                fp16_checks.append({"case": fp16_case, "shape": [m,n,k], "pattern": pattern,
                                    **validate_fp16_result(fp16_run, gpu_w, gpu_a), "kernel": dict(fp16_run["kernel"])})
                fp16_output = fp16_run["output"].cpu()
                if pattern == "random":
                    for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
                        measured = native._benchmark_mixed(fp16_case, mode, gpu_w, gpu_a, 0, 2, 10)
                        assert torch.equal(measured["output"].cpu().view(torch.int32), fp16_output.view(torch.int32))
                        assert measured["weight_cached"] == (mode in ("compute_only", "steady_state"))
                        assert measured["conversion_scope"] == "source_format_to_fp16"
                        fp16_checks.append({"case": fp16_case, "mode": mode, "timing_contract_version": measured["timing_contract_version"],
                                            "output_bitwise_equal": True})
                cw, wq, ws = convert(wsrc, f"{pattern}_w_{k}")
                ca, aq, asc = convert(asrc, f"{pattern}_a_{k}")
                expected = reference(aq, asc, wq, ws)
                from benchmark_a100_mixed import BINARY_TILES
                for btile in BINARY_TILES:
                    if btile.endswith("512") and k%512:
                        continue
                    for layout in ("row_major", "group_major"):
                        bv = "o9" if variant == "o7" else "o10"
                        modes = ("conversion_only", "compute_only", "cold", "steady_state") if pattern == "random" else ("compute_only",)
                        for mode in modes:
                            br = native._benchmark_mixed(bv, mode, move(wsrc), move(asrc), 0, 2, 10, btile, layout)
                            by = br["output"].cpu()
                            assert by.dtype == torch.float32 and torch.isfinite(by).all()
                            torch.testing.assert_close(by, expected, rtol=1e-3, atol=1e-3)
                            assert torch.equal(by.view(torch.int32), expected.view(torch.int32))
                            assert br["kernel"]["activation_planes"] == (8 if bv == "o9" else 6)
                            assert br["weight_cached"] == (mode in ("compute_only", "steady_state"))
                            assert br["total_timing"] == ("sum_of_batched_stage_samples" if mode == "conversion_only" else "single_execution_cuda_event")
                            binary_checks.append({"variant": bv, "tile": btile, "shape": [m,n,k], "pattern": pattern,
                                                  "layout": layout, "mode": mode, "reference_bitwise": True,
                                                  "mse_vs_paired_baseline": (by.double()-fp16_output.double()).square().mean().item(),
                                                  "kernel": dict(br["kernel"])})
                for tile in ("64x64x128", "64x128x256"):
                    output = native._benchmark_split_grouped(
                        ca["packed"], ca["scale"], cw["packed"], cw["scale"], 0, 1, tile)
                    y = output["output"].cpu()
                    assert y.dtype == torch.float32 and torch.isfinite(y).all()
                    torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)
                    gemm_checks.append({"variant": variant, "shape": [m, n, k],
                                        "pattern": pattern, "tile": tile,
                                        "max_abs_error_vs_fixed_reference": (y - expected).abs().max().item(),
                                        "mse_vs_fixed_reference": (y.double() - expected.double()).square().mean().item(),
                                        "mse_vs_synthetic_o0": (y.double() - o0.double()).square().mean().item(),
                                        "paired_baseline": fp16_case,
                                        "mse_vs_paired_baseline": (y.double() - fp16_output.double()).square().mean().item(),
                                        "o0_kernel": dict(o0_run["kernel"]),
                                        "kernel": dict(output["kernel"])})
                    # New physical scale layout must preserve each output bit.
                    layout_modes = ("conversion_only", "compute_only", "cold", "steady_state") if pattern == "random" else ("compute_only",)
                    for layout_mode in layout_modes:
                        major = native._benchmark_mixed(variant, layout_mode, move(wsrc), move(asrc),
                                                        0, 2, 10, tile, "group_major")
                        assert torch.equal(major["output"].cpu().view(torch.int32), y.view(torch.int32))
                        assert major["kernel"]["scale_layout"] == "group_major"
                        assert major["weight_cached"] == (layout_mode in ("compute_only", "steady_state"))
                        for key, ref in (("converted_weight", ws), ("converted_activation", asc)):
                            _, effective = major[key]
                            assert torch.equal(effective.cpu().contiguous().view(torch.int32), ref.view(torch.int32))
                        layout_checks.append({"variant": variant, "shape": [m, n, k], "pattern": pattern,
                                              "tile": tile, "mode": layout_mode, "bitwise_equal_row_major": True,
                                              "mse_vs_synthetic_o0": (y.double() - o0.double()).square().mean().item()})
                if pattern == "random":
                    for mode in ("conversion_only", "compute_only", "cold", "steady_state"):
                        sample = native._benchmark_mixed(variant, mode, move(wsrc), move(asrc),
                                                          2, 3, 10, "64x128x256")
                        torch.testing.assert_close(sample["output"].cpu(), expected, rtol=1e-3, atol=1e-3)
                        for key, reference_pair in (("converted_weight", cw), ("converted_activation", ca)):
                            packed, scale = sample[key]
                            assert torch.equal(packed, reference_pair["packed"])
                            assert torch.equal(scale.view(torch.int32), reference_pair["scale"].view(torch.int32))
                        expected_stages = {
                            "conversion_only": {"weight_conversion", "activation_conversion", "total"},
                            "compute_only": {"gemm", "total"},
                            "cold": {"weight_conversion", "activation_conversion", "gemm", "total"},
                            "steady_state": {"activation_conversion", "gemm", "total"},
                        }[mode]
                        assert set(sample["timings_ms"]) == expected_stages
                        for stage, times in sample["timings_ms"].items():
                            assert len(times) == 3 and all(t > 0 for t in times)
                            inner = 10 if "conversion" in stage or (stage == "total" and mode == "conversion_only") else 1
                            assert sample["stage_timing_inner_repeats"][stage] == inner
                        assert sample["weight_cached"] == (mode in ("compute_only", "steady_state"))
                        assert sample["activation_prepared"] == (mode == "compute_only")
                        assert sample["timing_contract_version"] == 2
                        assert sample["timing_strategy"] == "conversion_amortized_end_to_end_direct"
                        assert sample["measurement_order"] == "direct_path_then_isolated_conversions"
                        expected_method = "sum_of_batched_stage_samples" if mode == "conversion_only" else "single_execution_cuda_event"
                        assert sample["total_timing"] == expected_method
                        if mode == "conversion_only":
                            times = sample["timings_ms"]
                            # Native FP32 sum, not median(W)+median(A).
                            summed = torch.tensor(times["weight_conversion"]) + torch.tensor(times["activation_conversion"])
                            assert torch.equal(torch.tensor(times["total"]), summed)
                        if mode == "compute_only":
                            assert sample["timings_ms"]["total"] == sample["timings_ms"]["gemm"]
                        timing_checks.append({"variant": variant, "mode": mode, "shape": [m, n, k],
                                              "timings_ms": dict(sample["timings_ms"]),
                                              "stage_timing_inner_repeats": dict(sample["stage_timing_inner_repeats"]),
                                              "total_timing": sample["total_timing"],
                                              "timing_contract_version": sample["timing_contract_version"],
                                              "measurement_order": sample["measurement_order"],
                                              "weight_cached": sample["weight_cached"],
                                              "activation_prepared": sample["activation_prepared"]})

    rejected = []
    for fmt in mf.FORMATS:
        base = mf.quantize_source(torch.ones((2, 256)), fmt)
        for case in ("cpu", "nan_scale", "bad_shape", "wrong_dtype"):
            source = dict(base) if case == "cpu" else move(base)
            if case == "nan_scale":
                source["scale"] = torch.full_like(source["scale"], 255)
            elif case == "bad_shape":
                source["shape"] = [2, 128]
            elif case == "wrong_dtype":
                source["payload"] = source["payload"].float()
            try:
                native._convert_mixed_source(source)
            except (ValueError, RuntimeError):
                rejected.append(f"{fmt}/{case}")
            else:
                raise AssertionError(f"accepted invalid input: {fmt}/{case}")
    overflow = move(mf.quantize_source(torch.ones(2, 256), "mxfp8_e4m3_g128"))
    overflow["scale"].fill_(253)
    try:
        native._convert_mixed_source(overflow)
    except RuntimeError as exc:
        assert "scale overflow" in str(exc)
        rejected.append("mxfp8/fixed_scale_overflow")
    else:
        raise AssertionError("accepted fixed-scale overflow")
    report = {"passed": True, "scope": "synthetic_formats_conversion_and_integer_gemm",
              "formal_o7_o8_complete": False, "performance_acceptance": False,
              "data_source": "deterministic_synthetic_fp16_not_real_trace",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "codec_tests": result.testsRun,
              "conversion_checks": conversion_checks, "gemm_checks": gemm_checks,
              "fp16_checks": fp16_checks,
              "bitplane_checks": bitplane_checks,
              "binary_gemm_checks": binary_checks,
              "timing_contract_checks": timing_checks,
              "group_major_layout_checks": layout_checks,
              "rejected": rejected}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "scope": report["scope"],
                      "codec_tests": result.testsRun, "conversion_checks": len(conversion_checks),
                      "fp16_checks": len(fp16_checks),
                      "bitplane_checks": len(bitplane_checks),
                      "binary_gemm_checks": len(binary_checks),
                      "gemm_checks": len(gemm_checks), "timing_checks": len(timing_checks), "rejected": len(rejected),
                      "group_major_layout_checks": len(layout_checks),
                      "max_abs_error_vs_fixed_reference": max(x["max_abs_error_vs_fixed_reference"] for x in gemm_checks)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Synthetic codec -> native fixed conversion -> dual-INT4 correctness acceptance.

Does not read/rewrite the user's prepared trace, publish formal O5/O6 MSE, or
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
    from adangel.quantization import mixed_formats as mf
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4
    from validate_a100_split_grouped import reference
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires A100 SM80")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(repo / "tests/unit"), "test_mixed_formats.py")
    with (args.output / "codec_tests.txt").open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise AssertionError("codec tests failed or skipped; inspect codec_tests.txt")
    conversion_checks = []

    def move(source):
        return {key: value.cuda() if isinstance(value, torch.Tensor) else value
                for key, value in source.items()}

    def convert(source, label):
        q, scale = mf.to_fixed_reference(source)
        weight = mf.FORMATS[source["format"]][1] == 4
        packed = mf._pack_nibbles(q.to(torch.uint8) & 15) if weight else split_int8_to_packed_int4(q)
        # Transfer, conversion and consumers are ordered on the same non-default stream.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            device_source = move(source)
            converted = native._convert_mixed_source(device_source)
        stream.synchronize()
        torch.testing.assert_close(converted["packed"].cpu(), packed, rtol=0, atol=0)
        assert torch.equal(converted["scale"].cpu().view(torch.int32), scale.view(torch.int32))
        conversion_checks.append({"case": label, "format": source["format"],
                                  "shape": source["shape"], "packed_bitwise_equal": True,
                                  "scale_bitwise_equal": True})
        return converted, q, scale

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

    gemm_checks = []
    for m, n, k in ((64, 128, 256), (128, 128, 512), (64, 128, 4096)):
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
            o0 = native.benchmark_o0(ai.cuda(), asc.cuda(), wm.cuda(), wms.cuda(),
                                     "compute_only", 0, 1, 1)["output"].cpu()
            for variant, (wfmt, afmt) in mf.VARIANTS.items():
                cw, wq, ws = convert(mf.quantize_source(w, wfmt), f"{pattern}_w_{k}")
                ca, aq, asc = convert(mf.quantize_source(a, afmt), f"{pattern}_a_{k}")
                expected = reference(aq, asc, wq, ws)
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
                                        "kernel": dict(output["kernel"])})

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
    report = {"passed": True, "scope": "synthetic_formats_conversion_and_integer_gemm",
              "formal_o5_o6_complete": False, "performance_acceptance": False,
              "data_source": "deterministic_synthetic_fp16_not_real_trace",
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "codec_tests": result.testsRun,
              "conversion_checks": conversion_checks, "gemm_checks": gemm_checks,
              "rejected": rejected}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "scope": report["scope"],
                      "codec_tests": result.testsRun, "conversion_checks": len(conversion_checks),
                      "gemm_checks": len(gemm_checks), "rejected": len(rejected),
                      "max_abs_error_vs_fixed_reference": max(x["max_abs_error_vs_fixed_reference"] for x in gemm_checks)}, indent=2))


if __name__ == "__main__":
    main()

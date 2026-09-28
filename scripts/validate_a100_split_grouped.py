#!/usr/bin/env python3
"""Validate the prepared dual-G128 integer core, NOT O5/O6 quantization."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def summary(values):
    mean = statistics.mean(values)
    return {"count": len(values), "median_ms": statistics.median(values),
            "mean_ms": mean,
            "cv_percent": 100 * statistics.pstdev(values) / mean if mean else 0.0,
            "min_ms": min(values), "max_ms": max(values)}


def pack_q4(w):
    raw = w.to(dtype=__import__("torch").uint8) & 15
    return (raw[:, 0::2] | (raw[:, 1::2] << 4)).contiguous()


def reference(a, asc, w, wsc):
    """Exact integer group dots; emulate rounded scale and ordered FP32 FMA.

    CPU FP64 represents the small integer dot exactly. Scales in the test
    corpus have modest exponents, so FP64 also models the FP32 FMA reference.
    This is not an arbitrary-exponent correctly-rounded FP emulator.
    """
    import torch
    a, w = a.cpu().double(), w.cpu().double()
    asc, wsc = asc.cpu(), wsc.cpu()
    y = torch.zeros((a.shape[0], w.shape[0]), dtype=torch.float32)
    for g in range(a.shape[1] // 128):
        sl = slice(g * 128, (g + 1) * 128)
        p = a[:, sl] @ w[:, sl].T
        scale = (asc[:, g, None] * wsc[None, :, g]).float()
        y = (p * scale.double() + y.double()).float()
    return y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--large", action="store_true")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=200)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory")
    if args.warmup < 0 or args.repeats < 1:
        p.error("Invalid timing repetitions")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.arbitrary_bits import split_int8_to_packed_int4
    from adangel.quantization.mxfp4 import decode_ue8m0_tensor
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("Requires A100 SM80")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    records = []

    def run(a, asc, w, wsc, tile, warmup=0, repeats=1):
        return native._benchmark_split_grouped(
            split_int8_to_packed_int4(a), asc, pack_q4(w), wsc,
            warmup, repeats, tile)

    shapes = [(64, 64, 128), (64, 128, 256), (128, 256, 512),
              (64, 128, 768), (128, 128, 4096)]
    for m, n, k in shapes:
        for pattern in ("random", "zero_a", "zero_w", "extrema", "q6", "zero_scale"):
            torch.manual_seed(7056 + k)
            a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
            w = torch.randint(-8, 8, (n, k), device="cuda", dtype=torch.int8)
            groups = k // 128

            def scales(rows, multiplier):
                r = torch.arange(rows, device="cuda")[:, None]
                g = torch.arange(groups, device="cuda")[None, :]
                # Unique row/column/group pattern; includes non-power-of-two.
                return ((1 + ((r * multiplier + g * 29) % 113) / 128)
                        * torch.exp2(((r + 3 * g) % 7 - 10).float())).contiguous()

            asc, wsc = scales(m, 13), scales(n, 17)
            if pattern == "zero_a":
                a.zero_()
            if pattern == "zero_w":
                w.zero_()
            if pattern == "extrema":
                a[:, ::2] = -128
                a[:, 1::2] = 127
                w[:, ::2] = -8
                w[:, 1::2] = 7
            if pattern == "q6":
                # INT6 is sign-extended, never zero-extended, into INT8.
                a.copy_((torch.arange(k, device="cuda") % 64 - 32).to(torch.int8))
            if pattern == "zero_scale":
                asc[:, ::2] = 0
                wsc[:, 1::2] = 0
            expected = reference(a, asc, w, wsc)
            for tile in ("64x64x128", "64x128x256"):
                if tile == "64x128x256" and (n % 128 or k % 256):
                    continue
                # Run on a non-default stream to test the current-stream binding.
                with torch.cuda.stream(torch.cuda.Stream()):
                    result = run(a, asc, w, wsc, tile)
                y = result["output"].cpu()
                assert y.dtype == torch.float32 and torch.isfinite(y).all()
                torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)
                records.append({"shape": [m, n, k], "pattern": pattern, "tile": tile,
                                "max_abs_error": (y - expected).abs().max().item(),
                                "mse_vs_integer_reference": (y.double() - expected.double()).square().mean().item(),
                                "kernel": dict(result["kernel"])})

    # Reduction to the existing O3 semantics when A scale is constant over K.
    torch.manual_seed(3003)
    a = torch.randint(-128, 128, (128, 512), device="cuda", dtype=torch.int8)
    mx = torch.randint(0, 256, (128, 256), device="cuda", dtype=torch.uint8)
    asc = torch.linspace(.0001, .07, 128, device="cuda")
    codes = torch.randint(116, 127, (128, 4), device="cuda", dtype=torch.uint8)
    old = native.benchmark("o3", "compute_only", a, asc, mx, codes, 0, 1, 1, "baseline")
    old_prod = native.benchmark("o3", "compute_only", a, asc, mx, codes, 0, 1, 1, "production")
    torch.testing.assert_close(old_prod["output"], old["output"], rtol=0, atol=0)
    for tile in ("64x64x128", "64x128x256"):
        new = native._benchmark_split_grouped(
            split_int8_to_packed_int4(a), asc[:, None].expand(-1, 4).contiguous(),
            old["converted_weight"], decode_ue8m0_tensor(codes), 0, 1, tile)
        assert torch.equal(new["output"].view(torch.int32), old["output"].view(torch.int32))
        records.append({"pattern": "old_o3_bitwise_regression", "tile": tile,
                        "bitwise_equal": True})

    split = split_int8_to_packed_int4(a)
    asg = asc[:, None].expand(-1, 4).contiguous()
    wsg = decode_ue8m0_tensor(codes)
    rejected = []
    for bad in ("negative", "nan", "inf", "shape", "noncontiguous", "dtype", "overflow"):
        invalid = asg.clone()
        if bad == "negative":
            invalid[0, 0] = -1
        elif bad == "nan":
            invalid[0, 0] = float("nan")
        elif bad == "inf":
            invalid[0, 0] = float("inf")
        elif bad == "shape":
            invalid = invalid[:, :3].contiguous()
        elif bad == "noncontiguous":
            invalid = invalid.T.contiguous().T
        elif bad == "dtype":
            invalid = invalid.half()
        else:
            invalid.fill_(torch.finfo(torch.float32).max)
        try:
            native._benchmark_split_grouped(split, invalid, old["converted_weight"],
                                            wsg, 0, 1, "64x128x256")
        except RuntimeError:
            rejected.append(bad)
        else:
            raise AssertionError(f"Input validation accepted {bad}")

    large = []
    if args.large:
        # Synthetic prepared-input performance: NOT formal O5/O6 or MSE vs O0.
        torch.manual_seed(7056)
        a = torch.randint(-128, 128, (4096, 4096), device="cuda", dtype=torch.int8)
        w = torch.randint(-8, 8, (4096, 4096), device="cuda", dtype=torch.int8)
        asc = (torch.rand((4096, 32), device="cuda") + .5) * .01
        wsc = (torch.rand((4096, 32), device="cuda") + .5) * .01
        split, packed = split_int8_to_packed_int4(a), pack_q4(w)
        expected = reference(a, asc, w, wsc)
        for tile in ("64x64x128", "64x128x256"):
            result = native._benchmark_split_grouped(
                split, asc, packed, wsc, args.warmup, args.repeats, tile)
            y = result["output"].cpu()
            assert torch.isfinite(y).all()
            torch.testing.assert_close(y, expected, rtol=1e-3, atol=1e-3)
            times = list(result["timings_ms"]["gemm"])
            stats = summary(times)
            large.append({"tile": tile, "summary": stats, "raw_ms": times,
                          "timing_stable_cv3": stats["cv_percent"] < 3.0,
                          "max_abs_error": (y - expected).abs().max().item(),
                          "mse_vs_integer_reference": (y.double() - expected.double()).square().mean().item(),
                          "kernel": dict(result["kernel"])})

    report = {"passed": True, "correctness_passed": True,
              "timing_stable_cv3": all(r["timing_stable_cv3"] for r in large) if large else None,
              "scope": "prepared_integer_core_only",
              "formal_o5_o6_complete": False,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "checks": records, "rejected": rejected,
              "synthetic_large": large}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(records), "rejected": rejected,
                      "synthetic_large": [{k: v for k, v in r.items() if k != "raw_ms"} for r in large],
                      "scope": report["scope"]}, indent=2))


if __name__ == "__main__":
    main()

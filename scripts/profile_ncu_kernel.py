#!/usr/bin/env python3
"""Launch one prepared production GEMM under Nsight Compute.

The script deliberately avoids software references and the other timing modes.
Use ``--launch-skip WARMUP --launch-count 1`` in ncu so that the first measured
production launch, rather than a warmup launch, is collected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NCU_KERNEL_FILTERS = {
    "o1": "regex:adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup",
    "o2": "regex:^device_kernel$",
    "o3": "regex:adangel_o3_split_tma_ws",
    "o4": "regex:adangel_o4_bitwise_tma_ws",
}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/prepared/llama2_7b_prefill_o0_o4"),
        help="prepared trace directory",
    )
    parser.add_argument("--sample-id", default="layer_00_q_proj")
    parser.add_argument("--variant", choices=tuple(NCU_KERNEL_FILTERS), required=True)
    parser.add_argument(
        "--implementation",
        default="production",
        help="internal O3 implementation key; other variants only accept production",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--conversion-inner-repeats", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def select_sample(manifest: dict, sample_id: str) -> dict:
    matches = [
        sample
        for sample in manifest.get("samples", [])
        if sample.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one manifest entry for {sample_id!r}, found {len(matches)}"
        )
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.warmup < 0 or args.repeats <= 0 or args.conversion_inner_repeats <= 1:
        raise SystemExit(
            "warmup must be non-negative, repeats must be positive, and "
            "conversion-inner-repeats must exceed one"
        )

    import torch

    from adangel.ops.dispatch import benchmark_variant
    from adangel.ops.extension import require_variant
    from adangel.trace.storage import (
        load_prepared,
        sha256_file,
        validate_manifest,
    )

    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        formal=True,
        require_arbitrary_bits=args.variant in {"o3", "o4"},
    )
    sample = select_sample(manifest, args.sample_id)
    sample_path = args.data / sample["file"]
    actual_sha256 = sha256_file(sample_path)
    if actual_sha256 != sample["sha256"]:
        raise ValueError(
            f"prepared sample checksum mismatch: expected {sample['sha256']}, "
            f"got {actual_sha256}"
        )

    inputs = load_prepared(sample_path, device="cuda")
    if inputs.shape != (4096, 4096, 4096):
        raise ValueError(f"NCU formal profile requires 4096^3, got {inputs.shape}")

    if args.implementation != "production" and args.variant != "o3":
        raise ValueError("--implementation is currently supported only for O3")
    if args.variant == "o3" and args.implementation != "production":
        native = require_variant("o3")
        payload = native._benchmark_o3_impl(
            args.implementation,
            "compute_only",
            inputs.A_int8,
            inputs.A_scale,
            inputs.W_mxfp4_g128,
            inputs.W_scale_g128,
            args.warmup,
            args.repeats,
            args.conversion_inner_repeats,
        )
    else:
        payload = benchmark_variant(
            inputs,
            args.variant,
            "compute_only",
            warmup=args.warmup,
            repeats=args.repeats,
            backend="native",
            conversion_inner_repeats=args.conversion_inner_repeats,
        )
    torch.cuda.synchronize()
    output = payload["output"]
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise RuntimeError("profiled production output must be finite FP32")

    timings = dict(payload["timings_ms"])
    if set(timings) != {"gemm", "total"}:
        raise RuntimeError(f"unexpected compute-only timing stages: {sorted(timings)}")
    if any(len(values) != args.repeats for values in timings.values()):
        raise RuntimeError("native backend did not return the requested repeat count")

    result = {
        "passed": True,
        "sample_id": inputs.sample_id,
        "shape": list(inputs.shape),
        "variant": args.variant,
        "implementation_requested": args.implementation,
        "mode": "compute_only",
        "warmup_launches": args.warmup,
        "measured_launches": args.repeats,
        "ncu_kernel_filter": NCU_KERNEL_FILTERS[args.variant],
        "ncu_launch_skip": args.warmup,
        "ncu_launch_count": 1,
        "timings_ms": timings,
        "kernel": dict(payload["kernel"]),
        "note": "NCU replay duration is diagnostic and does not replace formal CUDA Event latency",
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

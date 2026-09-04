#!/usr/bin/env python3
"""Compare CUDA 12.8/13.x O3 MMA lowering and estimate its instruction floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_KERNELS = {
    "u4s4": "adangel_o3_mma_micro_u4s4_c4",
    "s4s4": "adangel_o3_mma_micro_s4s4_c4",
    "s8s8": "adangel_o3_mma_micro_s8s8_c4",
}


def load(directory: Path) -> dict[str, object]:
    attribution = json.loads(
        (directory / "static_instruction_attribution.json").read_text(encoding="utf-8")
    )["kernels"]
    return {
        "directory": str(directory),
        "toolchain": (
            (directory / "toolchain.txt").read_text(encoding="utf-8", errors="replace")
            if (directory / "toolchain.txt").exists()
            else "not recorded"
        ),
        "kernels": {kind: attribution[symbol] for kind, symbol in REQUIRED_KERNELS.items()},
    }


def core_lowering_counts(kernel: dict[str, object]) -> dict[str, int]:
    helper = kernel["outlined_lowering_per_ptx_mma"]
    if not isinstance(helper, dict):
        raise ValueError("sub-byte kernel does not contain an outlined lowering helper")
    counts = helper["static"]
    return {name: int(counts[name]) for name in ("IMMA", "LOP3", "SHF", "IMAD")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cuda12_directory", type=Path)
    parser.add_argument("cuda13_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--production-dynamic-sass", type=int, default=2_559_000_000)
    parser.add_argument("--production-ms", type=float, default=2.034335970878601)
    parser.add_argument("--o1-ms", type=float, default=0.622136)
    args = parser.parse_args()

    cuda12 = load(args.cuda12_directory)
    cuda13 = load(args.cuda13_directory)
    comparisons: dict[str, object] = {}
    for kind in REQUIRED_KERNELS:
        old = cuda12["kernels"][kind]
        new = cuda13["kernels"][kind]
        comparisons[kind] = {
            "cuda12_logical_tops": old["logical_tops"],
            "cuda13_logical_tops": new["logical_tops"],
            "cuda13_over_cuda12": new["logical_tops"] / old["logical_tops"],
            "cuda12_median_ms": old["median_ms"],
            "cuda13_median_ms": new["median_ms"],
        }

    u4_core = core_lowering_counts(cuda12["kernels"]["u4s4"])
    s4_core = core_lowering_counts(cuda12["kernels"]["s4s4"])
    # Formal O3 executes two K64 PTX MMAs for each low/high path per G128.
    core_per_g128_consumer_warp = 2 * sum(u4_core.values()) + 2 * sum(s4_core.values())
    ctas = (4096 // 64) * (4096 // 32)
    consumer_warps = ctas * 16
    g128_groups = 4096 // 128
    irreducible_core_dynamic = core_per_g128_consumer_warp * consumer_warps * g128_groups
    fraction = irreducible_core_dynamic / args.production_dynamic_sass
    optimistic_floor_ms = args.production_ms * fraction
    target_ms = 2.0 * args.o1_ms

    result = {
        "cuda12": cuda12,
        "cuda13": cuda13,
        "comparison": comparisons,
        "formal_o3_instruction_floor": {
            "u4s4_core_per_ptx": u4_core,
            "s4s4_core_per_ptx": s4_core,
            "core_per_g128_consumer_warp": core_per_g128_consumer_warp,
            "ctas_4096": ctas,
            "consumer_warps_4096": consumer_warps,
            "g128_groups": g128_groups,
            "irreducible_core_dynamic_sass": irreducible_core_dynamic,
            "measured_production_dynamic_sass": args.production_dynamic_sass,
            "irreducible_fraction": fraction,
            "optimistic_linear_floor_ms": optimistic_floor_ms,
            "half_o1_target_ms": target_ms,
            "floor_meets_target": optimistic_floor_ms <= target_ms,
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

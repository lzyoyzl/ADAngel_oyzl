#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adangel.trace.collector import collect_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect the formal Llama-2-7B WikiText FP16 prefill trace."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="absolute local Llama-2-7B model directory; model downloads are disabled",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="single CUDA device, defaulting to the trace config (cuda:0)",
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = collect_trace(
        args.model,
        args.output,
        config,
        device=args.device,
    )
    print(f"collected formal raw trace at {output}")


if __name__ == "__main__":
    main()

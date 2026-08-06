#!/usr/bin/env python3
from __future__ import annotations

import argparse

from adangel.trace.prepare import prepare_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True, help="formal experiment YAML")
    parser.add_argument("--trace-config", required=True, help="external trace selection YAML")
    args = parser.parse_args()
    records = prepare_trace(args.input, args.output, args.config, args.trace_config)
    print(f"prepared {len(records)} samples")


if __name__ == "__main__":
    main()

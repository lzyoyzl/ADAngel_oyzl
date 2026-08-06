#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from adangel.trace.collector import collect_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    collect_trace(args.model, args.text, args.output, config)


if __name__ == "__main__":
    main()

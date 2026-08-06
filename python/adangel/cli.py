"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _doctor(args) -> None:
    from .ops.extension import native_status, require_native

    status = native_status()
    print(json.dumps({"available": status.available, "reason": status.reason, "capabilities": status.capabilities}, indent=2))
    if args.require_native:
        require_native()


def _show_config(args) -> None:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(yaml.safe_dump(config, sort_keys=False))


def _run(args) -> None:
    from .benchmark.runner import run_experiment

    path = run_experiment(
        args.config,
        args.data,
        args.output,
        sample_limit=args.samples,
        warmup_override=args.warmup,
        repeats_override=args.repeats,
    )
    print(path)


def _analyze(args) -> None:
    from .analysis import generate_report

    print(generate_report(args.run, args.output))


def _verify_layout(args) -> None:
    from .ops.extension import require_sm120_extension

    native = require_sm120_extension()
    result = dict(native.verify_layout())
    print(json.dumps(result, indent=2))
    if not result.get("passed", False):
        raise RuntimeError("SM120 microscale layout verification failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adangel")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect CUDA/GPU/native backend readiness")
    doctor.add_argument("--require-native", action="store_true")
    doctor.set_defaults(func=_doctor)

    show = commands.add_parser("show-config", help="parse and print a YAML config")
    show.add_argument("--config", required=True)
    show.set_defaults(func=_show_config)

    run = commands.add_parser("run", help="run the formal interleaved RTX 5090 experiment")
    run.add_argument("--config", required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--samples", type=int)
    run.add_argument("--warmup", type=int)
    run.add_argument("--repeats", type=int)
    run.add_argument("--require-native", action="store_true", help="kept for explicit scripts; formal run always requires native")
    run.set_defaults(func=_run)

    analyze = commands.add_parser("analyze", help="generate the four tables and figures")
    analyze.add_argument("--run", required=True)
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(func=_analyze)

    verify = commands.add_parser("verify-layout", help="run the single-warp SM120 scale-layout test")
    verify.add_argument("--require-native", action="store_true")
    verify.set_defaults(func=_verify_layout)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

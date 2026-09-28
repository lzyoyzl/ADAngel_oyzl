#!/usr/bin/env python3
"""Synthetic fixture for the trace runner; never reads the real prepared data."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

from benchmark_a100_mixed_trace import run_sample, summarize_trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("use a fresh output directory")
    import torch
    from adangel import _sm80 as native
    from adangel.quantization.int8 import quantize_int8_per_row
    from adangel.quantization.mxfp4 import quantize_mxfp4, mxfp4_to_q4_packed
    from adangel.trace.schema import PreparedInputs
    from adangel.trace.storage import sha256_file
    if torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires A100 SM80")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    rows = {}

    def append(name, data):
        rows.setdefault(name, []).append(data)
        with (args.output/name).open("a") as stream:
            stream.write(json.dumps(data, allow_nan=False)+"\n")

    setup = SimpleNamespace(warmup=2, repeats=3, inner=10, rounds=1,
                            scale_layouts=["row_major", "group_major"])
    records = []
    for i, (size, pattern) in enumerate(((256, "random"), (512, "zero"))):
        torch.manual_seed(5600+size)
        a, w = torch.randn(size, size, device="cuda").half(), torch.randn(size, size, device="cuda").half()
        if pattern == "zero":
            a.zero_(); w.zero_()
        ai, asc = quantize_int8_per_row(a)
        w32, s32 = quantize_mxfp4(w)
        w128, s128 = quantize_mxfp4(w, group_size=128)
        x = PreparedInputs(f"synthetic_fixture_{pattern}", ai, asc, w32, s32, w128, s128, mxfp4_to_q4_packed(w128))
        records.extend(run_sample(x, i, setup, native, append, "synthetic_fixture_not_real_trace"))
    assert len(records) == 56
    assert len(rows["source_provenance.jsonl"]) == 2
    assert len(rows["source_formats.jsonl"]) == 4
    assert all(r["input_policy"] == "prepared_o0_fp16_bridge_secondary_quantization" for r in records)
    assert all(r["bitwise_equal_validation"] and r["timing_contract_version"] == 2 for r in records)
    assert {r["mode"] for r in records} == {"conversion_only", "compute_only", "cold", "steady_state"}
    assert not list(args.output.glob("*.pt"))
    for validation in rows["validation.jsonl"]:
        for variant in ("o5", "o6"):
            base = validation["mse_vs_o0"][f"{variant}/64x128x256"]
            assert base == validation["mse_vs_o0"][f"{variant}/64x128x256/group_major"]
    report = {"passed": True, "scope": "synthetic_trace_runner_fixture", "real_data_read": False,
              "samples": 2, "records": len(records), "formal_experiment_complete": False,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "binary_sha256": sha256_file(Path(native.__file__)), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "checks": "bridge bitwise vs O0; hashes, operand error budgets, output MSE, four timing modes, two layouts",
              "summary": summarize_trace(records)}
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("passed", "scope", "real_data_read", "samples", "records")}))


if __name__ == "__main__":
    main()

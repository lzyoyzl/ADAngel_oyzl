#!/usr/bin/env python3
"""Synthetic fixture for the trace runner; never reads the real prepared data."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

from benchmark_a100_mixed_trace import (
    INPUT_POLICY, RAW_INPUT_POLICY, run_sample, summarize_trace, tensor_identity, verify_raw_prepared,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--large", action="store_true", help="also test a synthetic 4096^3 bridge")
    p.add_argument("--original-fp16", action="store_true", help="test direct original-FP16 source selection")
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
    fixtures = [(256, "random"), (512, "zero")]
    if args.large:
        fixtures.append((4096, "random_large"))
    for i, (size, pattern) in enumerate(fixtures):
        torch.manual_seed(5600+size)
        a, w = torch.randn(size, size, device="cuda").half(), torch.randn(size, size, device="cuda").half()
        if pattern == "zero":
            a.zero_(); w.zero_()
        qa, qw = (a.cpu(), w.cpu()) if args.original_fp16 else (a, w)
        ai, asc = quantize_int8_per_row(qa)
        w32, s32 = quantize_mxfp4(qw)
        w128, s128 = quantize_mxfp4(qw, group_size=128)
        x = PreparedInputs(f"synthetic_fixture_{pattern}", *[
            t.cuda() for t in (ai, asc, w32, s32, w128, s128, mxfp4_to_q4_packed(w128))])
        original = (qa, qw) if args.original_fp16 else None
        if args.original_fp16 and size == 256:
            for bad in ((qa.float(), qw), (qa[:, :-1], qw), (qa+1, qw)):
                try:
                    verify_raw_prepared(x, bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid/mismatched raw operands were accepted")
        records.extend(run_sample(x, i, setup, native, append, "synthetic_fixture_not_real_trace", original))
        if args.original_fp16:
            provenance = rows["source_provenance.jsonl"][-1]
            assert provenance["raw_prepared_replay_bitwise"]
            assert provenance["activation_source_input"] == tensor_identity(qa)
            assert provenance["weight_source_input"] == tensor_identity(qw)
            if pattern != "zero":
                assert provenance["weight_source_input"] != provenance["weight_bridge"]
    assert len(records) == 52*len(fixtures)
    assert len(rows["source_provenance.jsonl"]) == len(fixtures)
    assert len(rows["source_formats.jsonl"]) == 2*len(fixtures)
    expected_policy = RAW_INPUT_POLICY if args.original_fp16 else INPUT_POLICY
    assert all(r["input_policy"] == expected_policy for r in records)
    assert all(r["bitwise_equal_validation"] and r["timing_contract_version"] == 2 for r in records)
    assert {r["mode"] for r in records} == {"conversion_only", "compute_only", "cold", "steady_state"}
    assert not list(args.output.glob("*.pt"))
    for validation in rows["validation.jsonl"]:
        for variant in ("o7", "o8", "o9", "o10"):
            base = validation["mse_vs_o0"][f"{variant}/64x128x256"]
            assert base == validation["mse_vs_o0"][f"{variant}/64x128x256/group_major"]
    report = {"passed": True, "scope": "synthetic_trace_runner_fixture", "real_data_read": False,
              "input_policy_tested": expected_policy,
              "samples": len(fixtures), "shapes": [size for size, _ in fixtures],
              "records": len(records), "formal_experiment_complete": False,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "binary_sha256": sha256_file(Path(native.__file__)), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "checks": "bridge bitwise vs O0; hashes, operand error budgets, output MSE, four timing modes, two layouts",
              "summary": summarize_trace(records)}
    (args.output/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("passed", "scope", "real_data_read", "samples", "records")}))


if __name__ == "__main__":
    main()

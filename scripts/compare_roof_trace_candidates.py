#!/usr/bin/env python3
"""Compare two candidates within one completed, balanced real-trace run.

CPU-only aggregation, not an independent GPU correctness test. Retains every
record, including CV failures. Collapse rounds within each sample before the
descriptive sample bootstrap; never divide unrelated aggregate medians.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("roof_compare_metrics", ROOT / "python/adangel/benchmark/metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def compare(rows, reference, candidate, samples, rounds, variants, modes, allow_reassociation=False):
    if reference == candidate or samples < 1 or rounds < 1:
        raise ValueError("distinct candidates and positive coverage required")
    selected = [r for r in rows if r["tune"] in (reference, candidate)]
    index = {(r["sample_id"], r["variant"], r["mode"], r["round"], r["tune"]): r for r in selected}
    ids = sorted({r["sample_id"] for r in selected})
    expected = set(itertools.product(ids, variants, modes, range(rounds), (reference, candidate)))
    if len(index) != len(selected) or len(ids) != samples or set(index) != expected:
        raise ValueError("incomplete or duplicate sample/variant/mode/round/tune coverage")
    for r in selected:
        if r['tune']==37 and not r.get('bitwise_equal_dynamic_tree'):
            raise ValueError('missing bitwise comparison against dynamic tree36')
        if r['tune'] in (39,40) and not r.get('bitwise_equal_pair_tree'):
            raise ValueError('missing bitwise comparison against pair tree38')
        if allow_reassociation and r['tune'] in (24,25,26,27,34,35,36,37,38,39,40):
            required=('mse_vs_production','max_abs_vs_production','mse_vs_semantic_fp64','max_abs_vs_semantic_fp64')
            if (not r.get('fp32_reassociated') or not r.get('semantic_tolerance_passed')
                    or not r.get('mse_regression_passed')
                    or any(not math.isfinite(r[key]) or r[key]<0 for key in required)):
                raise ValueError('reassociation numerical acceptance incomplete/failed')
        elif not r["bitwise_equal_production"] or r["mse_vs_production"] != 0:
            raise ValueError("candidate failed recorded bitwise correctness")
        value = r["summary"]["median_ms"]
        if not (0 < value < float("inf")):
            raise ValueError("invalid latency")
    result = []
    for variant, mode in itertools.product(variants, modes):
        per_sample = []
        for sid in ids:
            a = [index[sid, variant, mode, r, reference] for r in range(rounds)]
            b = [index[sid, variant, mode, r, candidate] for r in range(rounds)]
            groups=(a,b) if allow_reassociation else (a+b,)
            if any(len({(r["mse_vs_o0"], r["mse_vs_paired_fp16"]) for r in group})!=1 for group in groups):
                raise ValueError("reference MSE differs across implementations or rounds")
            per_sample.append(dict(sample_id=sid,
                reference_median_ms=statistics.median(r["summary"]["median_ms"] for r in a),
                candidate_median_ms=statistics.median(r["summary"]["median_ms"] for r in b),
                paired_speedup=statistics.median(x["summary"]["median_ms"] / y["summary"]["median_ms"] for x, y in zip(a, b)),
                mse_vs_o0=b[0]["mse_vs_o0"], mse_vs_paired_fp16=b[0]["mse_vs_paired_fp16"],
                reference_mse_vs_o0=a[0]['mse_vs_o0'],reference_mse_vs_paired_fp16=a[0]['mse_vs_paired_fp16'],
                candidate_mse_vs_production=b[0]['mse_vs_production']))
        speeds = [r["paired_speedup"] for r in per_sample]
        cv = {}
        for tune in (reference, candidate):
            group = [r for r in selected if (r["variant"], r["mode"], r["tune"]) == (variant, mode, tune)]
            cv[str(tune)] = dict(records=len(group), selected_stage_failed=sum(r["summary"]["cv_percent"] >= 3 for r in group),
                any_stage_failed=sum(any(v["cv_percent"] >= 3 for v in r["stage_summaries"].values()) for r in group))
        result.append(dict(variant=variant, mode=mode, samples=samples, rounds=rounds,
            reference_median_ms=statistics.median(r["reference_median_ms"] for r in per_sample),
            candidate_median_ms=statistics.median(r["candidate_median_ms"] for r in per_sample),
            paired_speedup_median=statistics.median(speeds),
            paired_speedup_ci95=list(metrics.bootstrap_median_ci(speeds, 10000, .95, 20260930)) if samples > 1 else None,
            median_mse_vs_o0=statistics.median(r["mse_vs_o0"] for r in per_sample),
            median_mse_vs_paired_fp16=statistics.median(r["mse_vs_paired_fp16"] for r in per_sample),
            median_reference_mse_vs_paired_fp16=statistics.median(r['reference_mse_vs_paired_fp16'] for r in per_sample),
            median_candidate_mse_vs_production=statistics.median(r['candidate_mse_vs_production'] for r in per_sample),
            cv=cv, per_sample=per_sample))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=int, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument('--allow-reassociation',action='store_true')
    args = parser.parse_args()
    paths = [args.input / name for name in ("environment.json", "summary.json", "results.jsonl")]
    data = [p.read_bytes() for p in paths]
    env, summary = (json.loads(value) for value in data[:2])
    rows = [json.loads(line) for line in data[2].splitlines()]
    run_args = env["args"]
    if not summary.get("correctness_passed") or not summary.get("no_filtering"):
        raise ValueError("completed correctness summary required")
    if args.allow_reassociation and (not run_args.get('allow_reassociation') or
            summary.get('numerical_policy') not in ('explicit_reassociation_only_for_24_25','explicit_reassociation_only_for_24_25_26_27','explicit_reassociation_only_for_24_25_26_27_34_35_36','explicit_reassociation_only_for_24_25_26_27_34_35_36_37','explicit_reassociation_only_for_24_25_26_27_34_35_36_37_38','explicit_reassociation_only_for_24_25_26_27_34_35_36_37_38_39_40')):
        raise ValueError('run was not explicitly opted into reassociation')
    if bool(summary.get("all_four_modes_completed")) != bool(run_args["all_modes"]):
        raise ValueError("declared mode completion mismatch")
    modes = ("conversion_only", "compute_only", "cold", "steady_state") if run_args["all_modes"] else ("compute_only",)
    result = compare(rows, args.reference, args.candidate, run_args["samples"], run_args["rounds"], run_args["variants"], modes,args.allow_reassociation)
    print(json.dumps(dict(reference_tune=args.reference, candidate_tune=args.candidate,
        source_binary_sha256=env["binary_sha256"],
        reassociation_opt_in=args.allow_reassociation,
        sources=[dict(file=str(p), sha256=hashlib.sha256(b).hexdigest()) for p, b in zip(paths, data)],
        policy="same-run paired rounds; no CV filtering; sample bootstrap descriptive, same-trace samples correlated",
        rows=result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

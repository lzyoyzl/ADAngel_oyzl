"""Generate the four contract tables and figures from one completed run."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..benchmark.metrics import bootstrap_median_ci


def _records(run_dir: Path, stability_policy: str) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("run is empty")
    if stability_policy not in {"strict", "diagnostic"}:
        raise ValueError(f"unknown stability policy: {stability_policy}")
    if stability_policy == "strict" and any(not row["valid"] for row in rows):
        raise RuntimeError("run is empty or contains invalid/CV-failed records")
    return rows


def generate_report(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    stability_policy: str = "strict",
    diagnostic_note: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    table_dir, figure_dir = output_dir / "tables", output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    records = _records(run_dir, stability_policy)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    stability_rows = []
    for row in records:
        failed_stages = [
            stage
            for stage, stats in row["summary"].items()
            if float(stats["cv_percent"]) >= float(config["timing"]["max_cv_percent"])
        ]
        stability_rows.append(
            {
                "sample_id": row["sample_id"],
                "variant": row["variant"],
                "mode": row["mode"],
                "valid": bool(row["valid"]),
                "stable_cv": bool(row["stable_cv"]),
                "failed_stages": ",".join(failed_stages),
                "max_stage_cv_percent": max(
                    float(stats["cv_percent"]) for stats in row["summary"].values()
                ),
            }
        )
    pd.DataFrame(stability_rows).to_csv(table_dir / "00_stability.csv", index=False)
    unstable_records = sum(not row["valid"] for row in records)
    (output_dir / "report_metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(run_dir),
                "stability_policy": stability_policy,
                "clock_policy": (
                    "dynamic_boost_not_locked"
                    if stability_policy == "diagnostic"
                    else "strict_cv_gate"
                ),
                "records": len(records),
                "unstable_records": unstable_records,
                "cv_threshold_percent": float(config["timing"]["max_cv_percent"]),
                "diagnostic_note": diagnostic_note,
                "warning": (
                    "Timing medians include CV-failed records; inspect tables/00_stability.csv. "
                    "No records were filtered or replaced."
                    if unstable_records
                    else None
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    flat = []
    for row in records:
        base = {
            k: row[k]
            for k in (
                "sample_id",
                "variant",
                "mode",
                "mse_vs_o0",
                "equivalent_tflops",
                "valid",
                "stable_cv",
            )
        }
        for stage, stats in row["summary"].items():
            flat.append({**base, "stage": stage, **stats})
    frame = pd.DataFrame(flat)

    conversion_names = {
        "o0": ["weight_conversion", "activation_conversion"],
        "o1": ["weight_conversion"],
        "o2": ["weight_conversion", "activation_conversion"],
    }
    conv = frame[frame["mode"].eq("conversion_only")].copy()
    conv = conv[conv.apply(lambda row: row["stage"] in conversion_names[row["variant"]], axis=1)]
    conv.to_csv(table_dir / "01_conversion_overhead.csv", index=False)

    gemm = frame[(frame["mode"] == "compute_only") & (frame["stage"] == "gemm")].copy()
    baseline = gemm[gemm["variant"] == "o0"][["sample_id", "median_ms"]].rename(columns={"median_ms": "o0_median_ms"})
    gemm = gemm.merge(baseline, on="sample_id")
    gemm["speedup_vs_o0"] = gemm["o0_median_ms"] / gemm["median_ms"]
    gemm.to_csv(table_dir / "02_gemm_only.csv", index=False)

    e2e = frame[(frame["mode"].isin(["cold", "steady_state"])) & (frame["stage"].isin(["weight_conversion", "activation_conversion", "gemm", "total"]))]
    e2e.to_csv(table_dir / "03_end_to_end.csv", index=False)

    mse_rows = {}
    for row in records:
        if row["mode"] == "compute_only":
            mse_rows[(row["sample_id"], row["variant"])] = row["mse_vs_o0"]
    mse_frame = pd.DataFrame(
        [{"sample_id": key[0], "variant": key[1], "mse_vs_o0": value} for key, value in mse_rows.items()]
    )
    summary_rows = []
    for variant, group in mse_frame.groupby("variant"):
        values = group["mse_vs_o0"].tolist()
        lo, hi = bootstrap_median_ci(
            values,
            int(config["statistics"]["bootstrap_samples"]),
            float(config["statistics"]["confidence"]),
            int(config["experiment"]["seed"]),
        )
        summary_rows.append(
            {
                "sample_id": "__summary__",
                "variant": variant,
                "mse_vs_o0": group["mse_vs_o0"].median(),
                "iqr": group["mse_vs_o0"].quantile(0.75) - group["mse_vs_o0"].quantile(0.25),
                "max": group["mse_vs_o0"].max(),
                "bootstrap_median_ci_low": lo,
                "bootstrap_median_ci_high": hi,
            }
        )
    pd.concat([mse_frame, pd.DataFrame(summary_rows)], ignore_index=True).to_csv(table_dir / "04_mse.csv", index=False)

    figure_note = "" if stability_policy == "strict" else " (dynamic Boost; CV diagnostic)"
    ax = conv.groupby(["variant", "stage"])["median_ms"].median().unstack().plot.bar()
    ax.set_ylabel("Median conversion latency (ms)")
    ax.set_title("Conversion overhead" + figure_note)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "01_conversion_overhead.png", dpi=180)
    plt.close(ax.figure)

    ax = gemm.groupby("variant")["equivalent_tflops"].median().plot.bar()
    ax.set_ylabel("Equivalent throughput (TFLOP/s)")
    ax.set_title("GEMM-only performance" + figure_note)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "02_gemm_only.png", dpi=180)
    plt.close(ax.figure)

    breakdown = e2e[e2e["stage"] != "total"].groupby(["variant", "mode", "stage"])["median_ms"].median().unstack(fill_value=0)
    ax = breakdown.plot.bar(stacked=True)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("End-to-end breakdown" + figure_note)
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "03_end_to_end_breakdown.png", dpi=180)
    plt.close(ax.figure)

    plot_mse = mse_frame[mse_frame["variant"].isin(["o1", "o2"])]
    ax = plot_mse.boxplot(column="mse_vs_o0", by="variant", grid=False)
    ax.set_ylabel("MSE vs O0")
    ax.set_title("MSE vs O0" + figure_note)
    ax.figure.suptitle("")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "04_mse_boxplot.png", dpi=180)
    plt.close(ax.figure)
    return output_dir

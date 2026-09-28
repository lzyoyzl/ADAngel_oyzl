#!/usr/bin/env python3
"""Read NCU wide raw exports without mixing profiling and Event measurements."""
import argparse
import csv
import json
from pathlib import Path

METRICS = (
    "gpu__time_duration.sum",
    "gpc__cycles_elapsed.avg.per_second",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed.sum",
    "derived__memory_l1_wavefronts_shared_excessive",
    "derived__memory_l2_theoretical_sectors_global_excessive",
)


def read_single_kernel(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2:
        raise ValueError("expected units row plus exactly one profiled kernel")
    units, data = rows
    names = list(METRICS) + sorted(k for k in data if k.startswith(
        "smsp__average_warps_issue_stalled_") and k.endswith("per_issue_active.ratio"))
    return {"source": str(path), "kernel": data["Kernel Name"],
            "metrics": {k: {"value": float(data[k].replace(",", "")), "unit": units[k]}
                        for k in names}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    result = {"scope": "NCU_diagnostic_not_Event_benchmark",
              "notes": ["Values retain NCU's exported units; duration may be us, not ms.",
                        "Stall ratios are cycles per issued instruction, not runtime fractions.",
                        "Theoretical excessive sectors are not measured DRAM bytes."],
              "kernels": [read_single_kernel(p) for p in args.inputs]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

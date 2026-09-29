#!/usr/bin/env python3
"""Render a verified completed trace run as grouped Markdown (no filtered records)."""
import argparse
import json
from pathlib import Path
import statistics as st


def render(directory):
    config = json.loads((directory/"config.json").read_text())
    summary = json.loads((directory/"summary.json").read_text())
    environment = json.loads((directory/"environment.json").read_text())
    records = [json.loads(s) for s in (directory/"results.jsonl").read_text().splitlines() if s.strip()]
    if not summary["correctness_passed"] or not summary["raw_fp16_experiment"]:
        raise ValueError("requires completed original-FP16 trace run")
    selected = [r for r in records if r["case"].split("/")[0] in {"o5","o6","o7","o8","o9","o10"}]
    modes = {"conversion_only", "compute_only", "cold", "steady_state"}
    if {r["mode"] for r in selected} != modes:
        raise ValueError("four-mode report requires all four modes; do not relabel a compute-only replication")
    if len(selected) != config["samples"] * config["rounds"] * 6 * 4:
        raise ValueError("incomplete or multiple-implementation coverage")
    identities = {(r["sample_id"], r["case"], r["mode"], r["round"]) for r in selected}
    if len(identities) != len(selected) or len({r["sample_id"] for r in selected}) != config["samples"]:
        raise ValueError("duplicate or missing sample coverage")
    if any(not r["bitwise_equal_validation"] or r["input_policy"] != "original_fp16_direct_source_quantization" for r in selected):
        raise ValueError("requires verified original-FP16 records")
    if any(r["experiment_naming_version"] != 3 or r["timing_contract_version"] != 2 for r in selected):
        raise ValueError("mixed naming/timing versions")
    rows = {r["case"]: r for r in summary["records"] if r["mode"] == "compute_only"}
    cases = {v: sorted({r["case"] for r in selected if r["case"].split("/")[0]==v}) for v in ("o5","o7","o9","o6","o8","o10")}
    if any(len(v)!=1 for v in cases.values()):
        raise ValueError("select exactly one implementation per variant; cannot silently choose winners")
    cases = {v: c[0] for v,c in cases.items()}
    expected = {(sid,case,mode,round_id) for sid in {r["sample_id"] for r in selected}
                for case in cases.values() for mode in modes for round_id in range(config["rounds"])}
    if identities != expected:
        raise ValueError("missing case/mode/round coverage")
    lines = ["# A100 O5–O10 分组实验结果", "",
        f"来源：`{directory.as_posix()}`；{config['samples']} 个真实样本，{config['rounds']} 轮，"
        f"warmup={config['warmup']}，repeats={config['repeats']}，转换 inner={config['inner']}。",
        f"源码：`{environment['commit']}`；二进制 SHA256：`{environment['binary_sha256']}`。", "",
        "组 A：O5/O7/O9 共用 NVFP4-G128 权重和 MXFP8 E4M3-G128 激活；"
        "组 B：O6/O8/O10 共用 HiF4-G128 权重和实验 NV-style FP6 E2M3-G128 激活。",
        "初始源量化在计时外。组内分别是 FP16、双 INT4、Binary 路径；FP32 输出。", "",
        "Median：先对每次重复取中位数，再在同样本多轮间取中位数，最后在样本间取中位数。"
        "Mean：对重复/轮次/样本取均值。加速比按同样本同轮配对后汇总，不能用表中边际 median 相除替代。", ""]
    for mode, title in (("conversion_only","转换开销"),("compute_only","GEMM-only / Compute-only"),
                        ("cold","Cold 端到端"),("steady_state","Steady-state 端到端")):
        lines += [f"## {title}", "", "| 组 | 后端 | Median ms | Mean ms | 相对组内 FP16 配对加速比 | CV 超阈记录/全部记录 |",
                  "|---|---|---:|---:|---:|---:|"]
        for variant,case in cases.items():
            r = next(r for r in summary["records"] if r["case"]==case and r["mode"]==mode)
            lines.append(f"| {'A' if variant in ('o5','o7','o9') else 'B'} | {variant.upper()} | {r['median_ms']:.6f} | {r['mean_ms']:.6f} | {r['paired_speedup_median']:.3f}× | {r['cv_failed_records']}/{r['records']} |")
        lines.append("")
        if mode == "conversion_only":
            lines += ["转换 total 是 W/A 隔离批量摊销样本逐次相加，不是联合 Event。单项开销：", "",
                      "| 后端 | W Median ms | W Mean ms | A Median ms | A Mean ms |", "|---|---:|---:|---:|---:|"]
            for variant,case in cases.items():
                rs=[r for r in selected if r["case"]==case and r["mode"]==mode]
                values=[]
                for stage in ("weight_conversion","activation_conversion"):
                    groups=[[r for r in rs if r["sample_id"]==sid] for sid in sorted({r['sample_id'] for r in rs})]
                    values += [st.median(st.median(r['summary'][stage]['median_ms'] for r in g) for g in groups),
                               st.fmean(st.fmean(r['summary'][stage]['mean_ms'] for r in g) for g in groups)]
                lines.append(f"| {variant.upper()} | "+" | ".join(f"{v:.6f}" for v in values)+" |")
            lines.append("")
    lines += ["## 输出 MSE（主参考为组内 FP16）", "",
              "| 后端 | 参考 | Median MSE | Mean MSE |", "|---|---|---:|---:|"]
    for variant,case in cases.items():
        r=rows[case]
        lines.append(f"| {variant.upper()} | {r['paired_baseline'].upper()} | {r['median_mse_vs_paired_baseline']:.10g} | {r['mean_mse_vs_paired_baseline']:.10g} |")
    lines += ["", "MSE 对 FP32 输出作 FP64 reduction；不是模型下游任务精度。相对旧 O0 的 MSE 另保留在 JSON，不与组内参考混用。", "",
              "## 实现标识", "", "| 后端 | 实际配置 |", "|---|---|"]
    for variant,case in cases.items():
        r=next(r for r in selected if r['case']==case)
        lines.append(f"| {variant.upper()} | `{case}`；`{r['kernel']['implementation']}` |")
    lines += ["", "没有删除离群值；CV 超过 3% 的记录如实列出。NCU 重放结果不混入 Event 性能表。",
              "Cold 包含 W/A 在线转换与 GEMM；steady-state 只缓存权重转换。各阶段独立实测，不能用 GEMM-only 加转换 median 推导端到端。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh report path")
    args.output.write_text(render(args.input),encoding="utf-8")

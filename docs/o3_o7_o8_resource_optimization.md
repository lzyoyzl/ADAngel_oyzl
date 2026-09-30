# O3/O7/O8 资源上界优化进展

更新：2026-09-30。目标：在保持量化、G128 scale 和累加语义的前提下，缩小 A100 实现与资源模型上界的差距。**目前仍在候选迭代，未达到约 0.282–0.286 ms 的模型下界，正式默认尚未切换。**

## 已验证的第一轮

保持 CTA `64×128×256`、两级 cp.async、两路原生 INT4 和 FP32 输出不变，将两个独立 N atom 的 MMA 交错发射。每个输出仍按原 G128 顺序重构、转 FP32、缩放和 FMA。

24 个真实样本，每样本 3 轮，warmup=50、repeats=200；同输入对照原正式 symbol。以下是准备好输入的 compute-only，不包含转换。

**顺序复核更正：**首轮脚本在两个实现时“轮换后反转”互相抵消，导致每种 variant 的先后顺序固定。正确性/MSE 仍有效，但下列几百分点收益可能受顺序影响，尚不能作为正式提升证明。已修正为明确的循环轮换并记录每条 execution_order，需重做平衡顺序配对；原始记录保留，不回填成新测量。

| 后端 | 原实现 median ms | 交错候选 median ms | 配对加速比 median | 配对加速比 95% CI | 候选相对原实现 MSE |
|---|---:|---:|---:|---:|---:|
| O3 | 0.558592 | 0.538624 | 1.043× | [1.036, 1.047] | 0 |
| O7 | 0.596992 | 0.567296 | 1.055× | [1.052, 1.058] | 0 |
| O8 | 0.600064 | 0.567296 | 1.059× | [1.052, 1.064] | 0 |

加速比先按同样本、同轮配对再汇总，因此不必等于两列总体 median 相除。所有候选输出与原实现逐位相同；相对参考的 MSE 也完全不变：

| 后端 | 相对 O0 的 median MSE | 组内 FP16 参考 | 相对组内参考的 median MSE |
|---|---:|---|---:|
| O3 | 0.0066530103 | O0 | 0.0066530103 |
| O7 | 0.0119003180 | O5 | 0.0055361724 |
| O8 | 0.0094403548 | O6 | 0.0044110849 |

ISA 审计确认同一候选函数包含 LDGSTS、`IMMA.16864.U4.S4` 和 `IMMA.16864.S4.S4`，无 INT8 替代。仍有少量 local load/store；审计按允许 spill 的策略通过，而非零 spill。

## 尚不能宣布正式完成的原因

收益约 4%–6%，离目标仍远。单独提前读取 scale 的初筛没有明显收益。计时也存在波动：72 条记录中，原实现/候选的 CV≥3% 数量分别为 O3 `64/12`、O7 `6/69`、O8 `57/13`。全部保留，不能据此宣称全阶段稳定验收；快照也不足以把波动归因于某个确定原因。

下一轮测试更宽的 N fragment，增加独立 MMA 链，同时检查额外寄存器压力；候选已接入原转换和四模式计时接口，默认仍是旧实现。后续还需内存安全、四模式真实数据及 NCU 复核。改动访存工作量后重新计算资源上界，不把旧上界当作性能承诺。

## 复现入口

重新编译 SM80 扩展后，使用新的输出目录：

```bash
python scripts/benchmark_a100_roof_candidates.py --synthetic --validate \
  --output runs/roof_screen_new --tunes -1 2 6 7
python scripts/audit_a100_o1.py --variant roof_candidate --allow-spills \
  --output reports/roof_audit_new
python scripts/benchmark_a100_roof_trace.py --tunes -1 2 \
  --output runs/roof_trace_new
# 四模式版本必须使用包含 roof_tune 参数的新编译产物；这是后续验收命令。
python scripts/benchmark_a100_roof_trace.py --all-modes --tunes -1 2 \
  --output runs/roof_four_modes_new
```

不需要重新采集 trace，不修改 5090 后端。记录、原始计时、来源 hash 和审计：[第一轮证据](evidence/a100_o378_roof_v1/README.md)。

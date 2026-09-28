# A100 O5/O6 计时对齐验收（v2）

源码与构建：`7760894`，2026-09-29。只修改 O5/O6 计时封装、新 A100 对照
脚本及说明；旧 O0–O4/SM120 后端、格式数值语义和 GEMM 内核没有修改。
这是合成输入的计时接口验收，不是 24 个真实样本的正式 O5/O6 结论。

## 对齐的内容

- 转换：每个 Event 区间执行 inner 次后除以 inner。
- conversion total：对齐旧 A100 O3，逐次相加 W/A 摊销样本，再统计。
  新对照脚本对 O0 使用同一统计口径，并另存其原生联合 Event total。
- compute-only：输入全部准备好，GEMM 单次直接计时。
- cold / steady-state：分别直接测一次 W+A+GEMM / A+GEMM；不相加阶段时间。
- Event 与缓冲区预热前分配；先直接路径，再独立转换；公共格式准备和 MSE
  在计时外。规则详见 [协议](../../o5_o6_a100_protocol.md)。

## 已通过的检查

| 检查 | 结果 |
|---|---|
| 计时/汇总单测 | 6 项通过，覆盖逐次求和、原始记录保留和禁止重构端到端时间 |
| 格式参考单测 | 15 项通过，无跳过 |
| CUDA 转换 | 51 项逐位对照通过 |
| GEMM | 36 项通过，对独立定点参考最大绝对误差和 MSE 均为 0 |
| 四模式接口 | 24 项通过，含 v2 标签、inner repeats、缓存语义与 total 求和检查 |
| 非法输入 | 17 项正确拒绝 |
| 同 kernel 指令审计 | 原生 U4×S4、S4×S4、cp.async/LDGSTS 通过，无 S8 替代 |

大 CTA 仍有既有的小量 spill（REG128、STACK8、2 LDL/2 STL）；审计按允许
spill 的策略通过，不是严格零 spill。小 CTA REG90、STACK0，无 local load/store。
本次没有重跑 Compute Sanitizer；数值内核没有变化，前次内存验证见
[格式验收](../a100_mixed_formats_v3/README.md)。

## 4096³ 四模式联调

`runs/mixed_synthetic_timing_v2/`：一个确定性 Gaussian FP16 合成样本，
O0/O3/O5/O6 四种后端 × 四模式，共 16 条记录；warmup50/repeats200/inner100，
一次交错轮次。O5/O6 均用 64×128×256 CTA。所有记录保存 v2 计时方法、
原生 total、原始样本和 MSE，四模式输出均与本次 compute-only 输出逐位一致。
O5/O6 的 4096³ 输出对独立定点参考误差为 0。

不是“所有 CV 均通过”：16 条中有 5 条存在 CV≥3% 的阶段，原始数据未删除或
替换。GPU 快照显示 SM 时钟在 1215–1410 MHz 间变化，快照中只有本测试计算
进程；这不等于连续独占监测，也不能逐个确定离群原因。

| 记录 | CV≥3% 的阶段 |
|---|---|
| O3 compute-only | GEMM、total |
| O5 cold | weight conversion、GEMM、total |
| O0 cold | weight conversion |
| O0 steady-state | activation conversion、GEMM、total |
| O5 steady-state | GEMM、total |

计时修改前后的 36 项小/中形状 GEMM 记录，形状、模式、参考误差以及 MSE vs
同源合成 O0 均完全一致。当前二进制 SHA-256：
`45f70a504a68295ff462d8bf79b989d68a6b1161ca0c53d4dabf8146d737a8e8`。
本次只是计时对齐与正确性验收，不代表 24 样本正式性能目标已完成。

## 原始证据

- `runs/mixed_formats_timing_v2/`：格式、转换、GEMM、四模式契约验证。
- `runs/mixed_synthetic_timing_v2/`：4096³ 四模式完整结果、元数据及 GPU 快照。
- `reports/audit_mixed_timing_v2/`：同二进制 ISA 和资源摘要；完整 PTX/SASS 留在 A100 同目录。
- `reports/build_mixed_timing_v2.log`：编译记录。
- `reports/mixed_timing_v2_unit_tests.txt`：6 项计时/汇总测试。
- `reports/mixed_synthetic_timing_v2_stdout.txt`：完整四模式运行摘要。

# O5–O10：v4 候选核对与 NCU

源码：`1108e5967947d8c902f4c0d28ccb88cde0f75328`。
二进制 SHA256：`36e59a1d34c226970ba0a5355c2988190b22f0d548c3e8813b32b7bd9477878c`。
本目录不是 24 样本四模式结果，不能与 [v3 正式记录](../a100_o5_o10_final_v3/README.md) 混成一轮。

| 检查 | 结果/文件 |
|---|---|
| 格式与数值 | `reports/o5_o10_formats_v4/validation.json`；15 codec、120 定点转换、102 FP16、120 bitplane、480 Binary GEMM、36 双 INT4 GEMM 等全部通过 |
| 实际函数审计 | Binary/INT4 的 `audit.json` 与 `resources.txt`；原生 BMMA 或 U4/S4 IMMA，加 LDGSTS；不是 probe |
| 新 swizzle 内存安全 | memcheck 4 项零错误，racecheck 4 项零 hazard；两格式×两布局 |
| 普通 Event | `runs/o5_o10_swizzle_screen_v4/`；合成 4096³、3 轮、warmup=50/repeats=200；原始记录与 CV 全保留 |
| NCU | `reports/ncu/o5_o10_v4/`；同二进制 O9 baseline/swizzle 各一正式 kernel；完整 raw CSV 和摘要 |

选定 Binary（非 swizzle、自然 scale、64×128×256）在此构建中：O9 REG=220、O10 REG=217，
STACK=0，未发现 LDL/STL。不同构建的寄存器分配可略有变化，不能拿 v3 的 REG 数冒充当前构建。
双 INT4 group-major 的 32-byte stack/少量 LDL/STL 仍如实报告；`--allow-spills`
只接受已批准的资源警告，不放宽原生指令检查。其他未选候选的资源警告也保留。

## Swizzle 的实际效果

只改变 shared word 地址 `word XOR (row & 4)`，不改输入、位权、G128、CTA 或量化语义。

| 指标 | 基线 | Swizzle |
|---|---:|---:|
| O9 普通 Event GEMM 中位 ms（三轮合成） | 2.581504 | 2.553856 |
| O10 普通 Event GEMM 中位 ms（三轮合成） | 2.045952 | 2.025472 |
| O9 NCU duration ms | 2.470400 | 2.424704 |
| NCU 平均 GPC GHz | 1.396872 | 1.404653 |
| 额外 shared wavefronts | 50,331,648 | 25,165,824 |
| L1/TEX throughput % | 21.427 | 14.855 |
| 动态 warp 指令数 | 834,412,544 | 838,836,224 |
| achieved occupancy % | 12.489 | 12.492 |

局部 shared 事务优化有效，但普通 Event 只改善约 1%，不能证明真实 24 样本稳定收益。
候选保留为内部选项 `64x128x256_swizzle`，不覆盖已完成的正式配置/结果。
Horner、16-warp 等之前候选见 [v3 preflight](../a100_o5_o10_preflight_v3/README.md)。

NCU 是 kernel replay、clock-control=none、cache-control=none、launch-skip=51/count=1。
九个 section：SpeedOfLight、ComputeWorkloadAnalysis、LaunchStats、Occupancy、SchedulerStats、
WarpStateStats、MemoryWorkloadAnalysis_Tables、InstructionStats、SourceCounters；不是 `--set full`。
精确 filter、skip 及运行环境见 `runs/ncu_o9_*_v4/profile_launch.json` 和 `environment.json`。
完整二进制 `.ncu-rep` 保存在 A100 项目目录 `reports/ncu/o5_o10_v4/`，本目录归档可读导出，
避免将大二进制报告加入 Git。SASS/PTX 全文同样保留在 A100 的对应 audit 目录。

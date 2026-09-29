# O5–O10 首版验收与候选初筛

这不是 24 样本最终性能验收。2026-09-29，A100-PCIE-40GB，CUDA 12.8，PyTorch 2.7.1+cu128。
CUDA 源码构建自 `6d9e0d5`，测试脚本为 `133f7c7`（后者未改 CUDA）。二进制 SHA256：
`920984b8a248c2c23a2e1fa23505bc478533a8ef2f1df1a84f89a3eb7267cd45`。

## 已通过的检查

- 复制后的 24 个原始 FP16 样本深度校验，raw/prepared manifest 来源一致。
- 15 项格式单测，102 项定点转换、102 项 bitplane 转换、93 项 FP16 检查。
- 192 项 Binary GEMM、36 项双 INT4 GEMM、72 项 group-major 回归；整数语义参考最大误差为 0。
- 256³、512³、4096³ 的原始 FP16 合成流程：156 条记录、四种计时、两个 scale 布局；Binary 与对应双 INT4 输出逐位一致。
- memcheck：0 errors；尚不以此替代后续 racecheck。
- 正式 Binary kernel 同时包含 BMMA AND/POPC 和 LDGSTS；双 INT4 kernel 包含 U4×S4、S4×S4 和 LDGSTS，不含 INT8 降级。
- 真实 `layer_00_q_proj` 的 36 条冒烟记录通过；不代表 24 样本统计。

## 4096³ 合成输入初筛（CUDA Event GEMM median）

50 次预热、200 次测量、3 轮、顺序轮换，未锁频、无离群过滤。表中为各轮 median 的中位数。

| 路径 | 配置 | ms |
|---|---|---:|
| O5 | FP16 基线 | 0.705536 |
| O7 | 64×128×256，group-major | 0.596992 |
| O9 | 64×128×256，row-major | 2.558976 |
| O6 | FP16 基线 | 0.732160 |
| O8 | 64×128×256，group-major | 0.604160 |
| O10 | 64×128×256，row-major | 2.065408 |

完整候选、各轮原始时延和 CV 见 `runs/o5_o10_cta_screen_v1/`。部分基线轮次 CV 超标，未隐藏，不把此表当作正式稳定收益声明。

## 初版 O9 的 NCU 诊断

来源 `reports/ncu/o5_o10_v1/o9_weighted_raw.csv`，9 个 sections，**不是 --set full**。
单个正式 `adangel_sm80_mixed_binary<8,128,256,0>`，跳过一次预执行＋50 次预热。
NCU 重放计时独立于上述 Event 表，不混用。

| 指标 | 值 |
|---|---:|
| NCU Duration | 2.432576 ms |
| SM throughput | 56.019506% |
| DRAM throughput | 2.272510% |
| 活跃 warp / 峰值 | 12.487228% |
| Eligible warp / scheduler active cycle | 0.864431 |
| 动态指令 | 822,091,776 |
| 额外 shared wavefronts | 50,331,648 |
| 寄存器/线程 | 218 |

证据支持继续减少整数重建指令和寄存器压力；不能仅据 DRAM 使用率低推断所有 memory stall 为零。具体 stall 比值保留原单位，不解释为运行时间百分比。

较大 tile 无 spill，但寄存器限制 occupancy。K512 有两种实例存在少量 LDL/STL，审计保留警告；未声称全部候选 zero-spill。原始 `.ncu-rep` 和完整 SASS 保留于 A100 同名目录。

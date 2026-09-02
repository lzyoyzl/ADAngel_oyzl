# O3 NCU 驱动性能优化迭代记录

本文记录 O3 在 RTX 5090（SM120a）上的每个性能优化候选。所有候选必须保持：

- `A8 = A_low_u4 + 16 * A_high_s4`；
- G128 独立 UE8M0 scale；
- U4×S4 与 S4×S4 INT4 Tensor Core 路径；
- INT32 partial 重构、FP32 scale/累加和 FP32 输出；
- 与 O0 比较的 MSE 不发生实现性变化；
- TMA、cooperative warp specialization、无全局 partial buffer；
- conversion-only、compute-only、cold、steady-state 四种计时口径不变。

生产实现只有在正确性、MSE、同入口指令审计、零 spill 和 24 样本配对性能同时通过后才切换。

## 基线：`n16_k128`

配置为 `128×16×128`、两级 TMA pipeline、1 producer warp 和 16 consumer warps。

正式 24 样本 compute-only median 为 `2.223352 ms`；NCU duration 为 `2.067 ms`。
当前 O1 compute-only median 为 `0.622136 ms`。若以 O1 一半的等效吞吐为方向，O3
需要接近 `2 × O1 latency ≈ 1.244 ms`。这是优化方向而不是允许改变数值语义的硬门槛。

NCU 基线证据：

| 指标 | 基线 |
|---|---:|
| Compute throughput | 91.36% |
| DRAM throughput | 1.63% |
| 动态 SASS | 约 2.80B |
| `LOP3+SHF+IMAD` | 约 2.175B（77.7%） |
| INT4 IMMA | 33.55M（约 1.2%） |
| Math-pipe throttle | 3.94 cycles/issued instruction |
| Shared excessive wavefronts | 251.66M（73.18%） |

判断：首要成本是 SM120 对 packed INT4 operand 的拆位、符号扩展和寄存器准备；第二成本是
packed row-major shared layout 的四路 bank conflict。DRAM、TMA、barrier 和 occupancy 不是首要瓶颈。

## Iteration 1：80-byte padded shared row

候选：

- `n16_k128_pad80`
- `n32_k128_pad80`

策略：packed K128 的逻辑行是 64 bytes，即 16 个 `uint32_t`。原 row stride 为 16 words，
consumer 的 `(lane_j,lane_i)` 访问映射到约四路 bank conflict。将物理 row stride 扩展为
20 words（80 bytes），使连续逻辑行的 bank 起点每次旋转 20，八个 row group 与四个 lane
column 覆盖32个bank。TMA仍只搬运64个逻辑bytes，额外16 bytes仅作为shared物理skew。

该候选不改变 packed 数据、INT4 PTX、MMA 数量、scale 或累加顺序。`N16` 隔离 padding
影响；`N32` 同时检查在冲突降低后扩大 N tile 是否能通过 A fragment 复用获得收益。

待服务器记录：

| 项目 | `n16_k128_pad80` | `n32_k128_pad80` |
|---|---:|---:|
| 小规模正确性 | 待测 | 待测 |
| 4096³ GEMM median | 待测 | 待测 |
| 相对基线配对速度 | 待测 | 待测 |
| MSE回归 | 待测 | 待测 |
| REG/STACK/LOCAL | 待审计 | 待审计 |
| Shared excessive wavefronts | 待NCU | 待NCU |
| 结论 | 待定 | 待定 |

## 后续候选准入顺序

1. 先验证 padding 是否实质降低 shared excessive wavefronts 和延迟。
2. 建立 packed INT4 单 warp lowering microbenchmark，区分 ptxas 必需展开与可消除的布局开销。
3. 若位操作主要来自可消除布局转换，再实现 MMA-native packed physical layout；若主要是
   SM120 对旧式 INT4 PTX 的固定 lowering，则如实记录硬件/ISA下界，不将显式 INT8 MMA
   冒充论文要求的 INT4 后端。
4. 之后再处理 group-major W scale、输出合并写回及 CTA/pipeline 复测。

每轮服务器数据必须附带命令、commit、GPU/driver、CUDA、CUTLASS commit、正确性、MSE、
四种计时、CV、PTX/SASS 同入口审计与资源使用；失败候选保留记录但不进入最终结果。

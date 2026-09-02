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

## Iteration 1：80-byte padded shared row（淘汰）

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

结果：CUDA 12.8/CUTLASS 可以编译该 layout，但 `n16_k128_pad80` 首次小规模运行即报告
`CUDA error: misaligned address`。原因是任意80-byte row stride不满足当前 TMA shared
descriptor/swizzle 的运行时对齐约束。候选在进入性能测量前淘汰，不改变 production。

## Iteration 2：TMA-native shared swizzle

候选：

- `n16_k128_swizzle`
- `n32_k128_swizzle`

用 CUTLASS `composition(Swizzle<2,4,3>, Layout<8×64,row-major>)` 构造 TMA B64
shared layout，再 tile 到完整 CTA/stage shape。物理容量仍为紧凑64 bytes/row；consumer
通过同一个 CuTe layout 计算每个 `uint32_t` fragment 地址。N16隔离swizzle，N32继续检查
A fragment复用。

最初尝试的 Ampere `Swizzle<3,3,3>` 被 CUTLASS TMA trait 在编译期拒绝；随后使用的
B128 `Swizzle<3,4,3>` 虽可运行，但其128-byte窗口与64-byte packed row不匹配，输出
错误。最终候选改为窗口一致的 B64 `Swizzle<2,4,3>`。该映射仍给八个 row group
产生 `0,16,4,20,8,24,12,28` 的bank起点。

第一次运行该 swizzle 后输出不匹配。CUTLASS 的 B128 swizzle 以1024-byte对齐的 shared
base 为寻址基准，而原 kernel 只声明128-byte动态shared对齐；Iteration 2.1 将 shared
base 提升到1024 bytes 后重新验证。

| 项目 | `n16_k128_swizzle` | `n32_k128_swizzle` |
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

## Iteration 2 服务器结果：B64 TMA swizzle（淘汰）

环境为 RTX 5090、CUDA 12.8 和固定 CUTLASS commit
`db1c288993354c88e551c40c19a8fb93a774a241`；候选源码 commit 为 `3e65078`。

两个 B64 候选均通过 `128³` 输出逐元素一致门槛（`max_abs_error=0`）。`4096³`
验证也通过，最大绝对误差与生产基线相同（`9.1552734375e-05`）。短时 CUDA Event
结果如下：

| Implementation | Compute-only median (ms) | CV (%) | 相对基线速度 |
|---|---:|---:|---:|
| `n16_k128` | 2.090496 | 0.377 | 1.000x |
| `n16_k128_swizzle` | 2.294432 | 0.257 | 0.911x |
| `n32_k128_swizzle` | 2.327200 | 0.510 | 0.898x |

定向 NCU memory pass 证明 swizzle 映射确实生效：

| Metric | Production | `n16_k128_swizzle` |
|---|---:|---:|
| NCU duration (ms) | 2.058336 | 2.273952 |
| Shared-load bank conflicts | 251,752,630 | 118,477 |
| Shared-load wavefronts | 344,027,318 | 92,393,165 |
| Executed shared loads | 88,080,384 | 88,080,384 |
| Registers/thread | 53 | 55 |

swizzle 几乎消除了被测 bank conflict，并将 shared-load wavefronts 降低 3.72 倍，
但 NCU 时间反而增加 10.5%。因此 shared conflict 是次要现象，主导成本仍是 SM120
对 packed INT4 的 lowering 和操作数准备。两个候选只作为内部负对照保留，不替换生产实现。

## Iteration 3：减小 M tile、提高 CTA 并发度

待测试候选：

- `m64_n16_k128`：CTA `64×16×128`，1 个 producer、8 个 consumer warp；
- `m64_n32_k128`：CTA `64×32×128`，1 个 producer、8 个 consumer warp，每个
  consumer warp 拥有两个 N replica。

目标是确认把 544-thread CTA 减至 288 threads 后，增加的 resident CTA/warp 调度自由度
能否隐藏较长的 packed-INT4 lowering 指令序列。TMA、两级 pipeline、G128 scale 语义、
两路 INT4 MMA、寄存器 partial、FP32 累加/输出和最终单次写回均保持不变。候选必须依次通过
小规模正确性、`4096³` 性能、MSE 回归、资源审计以及同一入口 TMA/INT4 指令审计才可晋升。

### Iteration 3 服务器短测结果

两个候选均通过 `128³` 逐元素一致验证，并在 `4096³` 上得到与基线相同的
`max_abs_error=9.1552734375e-05`：

| Implementation | Compute-only median (ms) | CV (%) | 相对基线速度 |
|---|---:|---:|---:|
| `n16_k128` | 2.090496 | 0.377 | 1.000x |
| `m64_n16_k128` | 2.081440 | 0.539 | 1.004x |
| `m64_n32_k128` | 2.146976 | 0.468 | 0.974x |

LaunchStats 表明基线为 544 threads、53 regs/thread，寄存器限制最多 2 CTA/SM，平均
active warps 为 69.82%；`m64_n16_k128` 为 288 threads、53 regs/thread，最多
4 CTA/SM，active warps 仅升至 73.68%。因此缩小 M tile 只产生约 0.4% 的微小短测收益，
扩大 N 还因每 warp 两个 N replica 的寄存器/串行工作而变慢。该结果不足以直接晋升，
`m64_n16_k128` 只保留到 24 样本配对阶段作为边界对照。

## Iteration 4：CuTe TiledMMA + LDSM fragment copy

候选 `n16_k128_cute_ldsm` 保持生产 CTA、TMA pipeline、G128 scale、U4×S4 与
S4×S4 INT4 MMA、INT32 重构及 FP32 累加不变。差别仅在 consumer 的操作数装载：

- 用 CuTe `TiledMMA` 建立 accumulator 和 A/B fragment 坐标；
- 用 `SM75_U32x4_LDSM_N` 从 shared memory 装载 MMA fragment；
- B fragment 在 low/high 两路 MMA 之间复用；
- 每 warp 只装载一份列 scale，再通过 shuffle 映射到拥有的输出元素。

目标是替换基线 NCU 中约 88.08M 条标量 shared load，同时避免手写 lane layout。
该候选首先只接受编译和小规模正确性检验；若 CuTe sub-byte fragment/layout 不匹配，
立即淘汰，不进入性能测量。

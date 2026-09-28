# O3/O5/O6：理论峰值与性能分析（阶段稿）

状态：O3 引用既有正式结果；O5/O6 格式已确认，尚未完成正式数据验收。
以下理论计算不代表实测加速比，也不承诺实现可达到理论下界。

## 1. 两路 INT4 的正确峰值口径

A100 标称 dense Tensor Core 峰值为 FP16 312 TFLOP/s、
INT8 624 TOPS、INT4 1248 TOPS；稀疏数字不能用于本实验。
这是 boost 时钟下的规格值，不是当前服务器持续运行的保证。
来源：[NVIDIA Ampere 架构说明，表 1](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。

令 D=2MNK，M=N=K=4096，则 D=137438953472 次等效算术操作。
一次 S8×S4 点积通过两路四位乘积实现时，Tensor Core 实际执行约 2D
次 INT4 算术操作；两路共享硬件发射/执行资源，不能各自独占整卡峰值。

| 路径 | Tensor Core 工作量 | 标称 dense 峰值 | 纯 MMA 理想下界 ms |
|---|---:|---:|---:|
| O0 FP16 | D | 312 TFLOP/s | 0.440510 |
| 单路 INT8（O1 的 MMA 部分） | D | 624 TOPS | 0.220255 |
| 双路 INT4（O3/O5/O6 的 MMA 部分） | 2D | 1248 TOPS | 0.220255 |

因此双路 INT4 的“有效 S8×S4 峰值”是 624 TOPS，不是 1248。
Q6 符号扩展为 INT8 后仍执行两路完整 INT4，不能按 6/8 比例减少指令成本。
只有改变算法并经过新审计，才能声称有效跳过某些运算。

以上下界忽略 scale、INT32→FP32、重构、数据搬运、同步、尾块、
寄存器资源、输出与调度等开销。它只是说明算力上的机会，不证明瓶颈所在。
用 D/t 报告等效吞吐；评估双路 INT4 的物理算术峰值利用率则用 2D/t/1248TOPS。
不能拿 D/t 直接除以 1248 来说“只有一半效率”。

## 2. 已有 O3 基线，不能冒充新 O5/O6

既有 A100 四模式实验，24 个真实样本的 compute-only median-of-medians：

| 后端 | GEMM ms | 与上述纯 MMA 下界之比 |
|---|---:|---:|
| O0 | 0.739328 | 1.68 |
| 优化 O1 | 1.143296 | 5.19 |
| 优化 O3 | 0.573440 | 2.60 |
| O5 | 待正式测试 | — |
| O6 | 待正式测试 | — |

来源：仓库 [A100 O3 验收记录](a100_o3_optimization.md)，
runs/a100_o3_production_four_modes_v25。它们是历史环境测量，
新后端是否超过 O0 必须重新做同进程、同输入来源、交错顺序配对。

O3 距纯算术下界约 2.60 倍，并不表示“还有可兑现的 2.60 倍加速”。
scale、输出、数据供应均非零。超过 O0 是合理待验证目标，不是已实现事实。

## 3. 新场景最应关注的成本

| 层次 | O3 | O5/O6 新增挑战 | 需要的证据 |
|---|---|---|---|
| 格式转换 | E2M1→Q4、INT8 split | 双侧浮点转定点；可能多级 scale/内部指数对齐 | 编解码穷举、饱和率、MSE 分解、conversion 时间 |
| scale | A 每行一个；W 每 G128 一个二次幂 | A/W 都逐 G128 变化，且可能非二次幂 | scale-load/后处理指令数、shared wavefront、采样热点 |
| MMA | 原生 U4×S4 与 S4×S4 | 保留两路，Q6 padding 不自动减少工作 | 同一正式函数 PTX+SASS |
| 数据供应 | swizzle、cp.async、N-slice fragment 复用 | 额外的双侧 scale 数据和同步依赖 | LDSM/LDGSTS、bank conflict、warp stall |
| 资源 | partial/FP32 acc 寄存器，允许经验证的小量 spill | scale 及 fragment 存活区间增加 | 寄存器、LOCAL/STACK、LDL/STL、occupancy |
| 全局写回 | 最终 FP32 输出一次 | 不应引入逐组完整输出读写 | 源码/SASS、DRAM/L2 事务、memcheck |

当前共用整数 kernel 已选择先复用 O3 的成熟路径，添加双侧 scale；
不能仅根据 SM throughput 高或 DRAM throughput 低就断言唯一瓶颈。
应结合 WarpStateStats、SchedulerStats、InstructionStats、SourceCounters
和 MemoryWorkloadAnalysis，在同一配置上验证假设。

## 4. 优化优先级与停止条件

先锁定数学语义与真实格式，再做同进程 64×64×128 / 64×128×256 配对。
如果 scale 后处理占比上升，优先考虑一组内的 row/column scale 寄存器复用、
scale panel 的读写合并和缩短存活区间；不得跨不同 G128 统一缩放。
若 shared/LDSM 压力高，检查 N-slice 的 A/W fragment 复用和 bank 行为；
若 MMA 依赖链占主导，再考虑有限的独立累加链，不为并行性无限增加寄存器。
只有实际被测为瓶颈的部分才继续优化。

每次候选保留原始计时、MSE、输出有限性、同函数 ISA 和内存安全证据。
无需锁频或等待整卡空闲；记录实际争用/时钟和离群值，不静默删除。
最终结论以真实 24 样本的配对结果为准；合成 prepared-core 测量只用于定位。

## 5. NCU 定位：双侧 scale 的物理布局

2026-09-29 对同源合成 4096³ 的 O3/O5/O6 单个正式 GEMM 做了 9-section
采集，未锁频，未刷新 cache。不是 `--set full`，也不是普通 Event 性能测试。
原始表和精确启动信息见 [NCU 证据](evidence/a100_mixed_ncu_v2/README.md)。
以下为尚未改 scale 布局的 row-major kernel：

| 指标 | O3 | O5 | O6 |
|---|---:|---:|---:|
| NCU duration ms（仅诊断） | 0.486976 | 0.544704 | 0.544064 |
| NCU GPC 平均时钟 GHz | 1.3025 | 1.3551 | 1.3542 |
| SM throughput % | 48.99 | 42.10 | 42.17 |
| DRAM throughput % | 12.48 | 10.88 | 11.06 |
| L1/TEX throughput % | 62.59 | 68.03 | 68.16 |
| 每 scheduler/cycle eligible warps | 0.709 | 0.820 | 0.820 |
| issue active % | 44.00 | 41.99 | 41.98 |
| achieved occupancy % | 24.25 | 24.28 | 24.31 |
| 动态 warp 指令数 M | 118.317 | 131.596 | 131.596 |

**首个有源码证据的改进点是 scale 的非合并加载，不是 DRAM 带宽不足。**
旧 FP32 scale 数组物理布局为 `[row,group]`。K4096 对应 32 个 group；
一次加载固定 group、相邻 row，因此相邻 lane 地址相差 32×4=128 字节。
warp 的 32 个 float 分散在 32 个 sector，而连续布局只需 4 个 sector。

O5 SourceCounters 中 `o3_optimized.cuh` 的两处 scale 加载：

| 源码操作（采集时行号） | 理论 global sectors | 理想 sectors | 额外 sectors |
|---|---:|---:|---:|
| A scale，80 行 | 4,194,304 | 524,288 | 3,670,016 |
| W scale，83 行 | 8,388,608 | 1,048,576 | 7,340,032 |
| 合计 | 12,582,912 | 1,572,864 | 11,010,048 |

两处合计覆盖该次 SourceCounters 的全部额外 global sectors。
这是按指令地址推算的事务量，不是实际 DRAM 字节数，不能直接换算为耗时。
同一报告的 shared wavefront 理论值与 ideal 都为 43,384,832，excessive=0；
因而“shared 操作多”不能在这里解释成“主要是 shared bank conflict”。

O5 WarpStateStats 中 MIO throttle 为 2.082、math pipe throttle 为 1.504、
wait 为 1.210、barrier 为 0.878（每条发射指令对应的平均等待周期口径）。
结合 issue active 约 42%、eligible warp 不足 1，可见访存发射、算术依赖和
同步均值得检查；MIO 采样热点还落在 LDGSTS，而不只在 scale 加载上。
这些状态不能各自解释成独占的 kernel 耗时比例，也不能直接相加预测加速比。
指标解释参见 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)。

针对性候选将 scale 改为物理 `[group,row]`，在转换 kernel 内直接写入，
GEMM 内连续读取；不增独立重排、不改两路 INT4 和 G128 累加顺序。
Python 逻辑 shape 仍为 `[row,group]`，stride 为 `[1,rows]`。
候选必须同时检查转换开销、四模式端到端、MSE、指令与内存安全；
仅凭 sector 减少不能宣布性能达标或替换旧默认。

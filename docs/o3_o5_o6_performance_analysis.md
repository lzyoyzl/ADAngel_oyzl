# O3/O5/O6：理论峰值与性能分析（阶段稿）

状态：O3 引用既有正式结果；O5/O6 尚未完成格式确认与真实 trace 验收。
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
| O5 | 待格式确认与正式测试 | — |
| O6 | 待格式确认与正式测试 | — |

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

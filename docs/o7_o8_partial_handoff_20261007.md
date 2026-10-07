# O7/O8：MMA 与整数后处理分工实验（v122）

2026-10-07，A100。**结果：候选正确但更慢，不替换当前最佳，也不迁移到 O3。**
源码在本地实现、推送 GitHub，再由 A100 项目 fetch/ff-only 后测试。正式扩展与 5090 代码未改。

## 1. 当前瓶颈与本轮针对性改动

当前最佳已经使用全 K 精确整数累加。逐 G128 的 FP32 转换/FMA 长链已不在快路径中，但整数 factor 加权、MMA 依赖及供数同步仍在。

此前预热后 O8 NCU 显示：eligible warps/scheduler 为 **0.7464**，issue active **46.90%**，Tensor pipeline **58.95%**；动态 IMAD **42.164M / 总指令105.521M**。这支持优先研究执行依赖与整数后处理，而不是继续只优化 HBM 带宽。该数据是单样本诊断，不是全24样本平均，详见[瓶颈与平台迁移报告](o3_o7_o8_bottleneck_portability_20261007.md)。

本轮不重复 v41 的“搬运 producer”，也不重复 v98/v104 的“八个 warp 都做 MMA”。保留原四个 warp 的 INT4 MMA 和 A/W fragment 复用，新增四个 warp 专做整数 scale/全 K 累加：

```text
4 个 MMA warp：原生两路 INT4 → INT32 G128 partial → 无损 INT16 打包
                                              ↓ shared 双缓冲
4 个后处理 warp：符号扩展回 INT32 → 原 Af×Wf 加权 → 全 K INT32 累加
                                              ↓
                                  原 FP32 基准 scale 恢复及一次输出
```

INT16 只是中间存储，不是再次量化。每个 CTA、每个 G128 用整数 Cauchy 上界证明：

```text
max_A_rows(sum(a²)) × max_W_rows(sum(w²)) ≤ 32767²
```

同时保留原全 K 溢出 guard。不满足 INT16 条件的整数 CTA 走已有宽 partial 路径；原需 FP32 回退的 CTA 继续回退。24 样本的最大保守 partial 上界：O7 **28,370**，O8 **10,010**；O7 全部 CTA、O8 每样本至少 **99.414%** CTA 可用快路径，O8 其余为原全 K guard 的回退。

## 2. 实际资源与指令审计

| 指标 | 原最佳 v78 | v122 候选 |
|---|---:|---:|
| CTA M×N×K | 64×128×128 | 64×128×128 |
| CTA warp 分工 | 4 个 warp 均做 MMA/后处理 | 4 MMA + 4 后处理 |
| Registers/thread | 168 | 128 |
| Shared/CTA（不含 driver 额外分配） | 34,304 B | 51,712 B |
| 实际可驻留 CTA/SM | 3 | 2 |
| 可驻留 warp/SM | 12 | 16，其中8个负责 MMA |
| 热 local load/store | 0 | 0 |
| 每 G128 MMA producer 的 S4×S4 / U4×S4 指令 | 32 / 32 | 32 / 32 |
| 每 G128 MMA producer 的 LDSM | 16 | 16 |
| 同工作量静态循环指令 | 383 | 271 producer +224 consumer =495（+29.24%） |

同一个候选 entry 确认为原生 `IMMA.16864.S4.S4` 与 `IMMA.16864.U4.S4`，不是 INT8 展开；原 v78 控制的完整 SASS 编码未变。

生产者/消费者采用独立 ready/empty named barrier；NVIDIA PTX 文档给出了这种 `bar.arrive` / `bar.sync` 交接方式，但其合法性不代表必然加速。[PTX 8.7 barrier 说明](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)

## 3. 全24样本配对实测

同进程、同源数据、单 stream、预分配、交错顺序；**24样本 ×3轮 ×200次测量，每轮预热1000次**。未锁频、未等待独占 GPU；不筛除离群值。

本轮仅测 **compute-only GEMM**。新增范围判断在计时外缓存，不能将本表写成 conversion、Cold 或 steady-state 成绩。

| Case | 原最佳 median ms | 候选 median ms | 配对吞吐变化 | 配对加速比95% CI | CV≥3%：原/新 |
|---|---:|---:|---:|---|---:|
| O7 | 0.478208 | 0.595968 | **−20.07%** | [0.79725, 0.80244] | 0/72；2/72 |
| O8 | 0.480256 | 0.601088 | **−20.24%** | [0.79643, 0.79932] | 0/72；1/72 |

配对加速比先计算每个样本三轮 old/new 的中位数，再汇总24样本；并非直接把本表两列总体中位数相除。CI 对24个配对样本 bootstrap 10,000次。少量 CV 失败记录全部保留；退化幅度大且 CI 完全小于1，不重复测量以追求更好的数字。

| Case / MSE 参考 | 原/新共同 MSE median | 原/新共同 MSE mean |
|---|---:|---:|
| O7 / O5 | 0.005536172273439442 | 0.005053635851002639 |
| O8 / O6 | 0.004411084910985704 | 0.004381379299073540 |

288条性能记录的输出均与旧最佳**逐位一致**且为 finite FP32，MSE 完全不变。独立随机、零值、饱和值、宽 INT32 回退、FP32 回退及非默认 stream 验证通过；针对该候选 entry 的 memcheck/synccheck 均0错误，racecheck 为0错误/0警告。有限测试不是对任意输入的形式化 GPU 安全证明。

## 4. 为什么不采纳，以及下一步的约束

这轮证明：**降低单线程寄存器数、提高总 resident warp，并不必然提高有效吞吐。**

原来留在同一线程寄存器里的 partial，现在要跨 warp 交接。即便无损压到 INT16，4096³ 全走快路径时，32个 G128 的 partial 仍额外产生 **2 GiB shared 读写**，还需要 packing/unpacking、factor 快照和握手同步。算术工作没有减少，循环静态指令反而增加29.24%。总 warp 增至16，但每SM负责 MMA 的 warp 从12个变为8个。

实测表明这些代价未被计算重叠抵消。这是结构证据与端到端 kernel 实测一致的解释；**本轮未再做 NCU，不能把其中某项单独认定为退化的唯一原因，也不能把静态指令增长当作延迟增长。**

因此停止此机制，不扫更多 warp 比例、handoff buffer 大小或 CTA；不追加其 conversion/E2E 测量，不迁移到逐组后处理更少的 O3。当前最佳仍为 O3 v89、O7/O8 v78+v73；O3本轮未重测。后续新方向必须能减少实际工作或在不新增大规模中间交接的前提下改善依赖，不能重复既有失败的 factor 缓存、链数、tile/stage 与系数预计算实验。

## 5. 是否可移植到其他平台？

**可移植的是算法原则，不是这次失败的 warp 配置或 A100 的速度结论。**

| 内容 | 可复用 | 需要按平台重做 |
|---|---|---|
| 全 K 整数对齐、guard、最终恢复 scale | 精确整数分解与范围证明方法 | 支持的数据格式、整数乘加范围、溢出回退及 MSE |
| 转换融合、权重缓存、寄存器 partial | 减少重复转换和中间读写 | packing、向量宽度、fragment 坐标、寄存器资源 |
| 多条独立 MMA chain、搬运/计算重叠 | 隐藏依赖的调度思路 | 原生 MMA、copy 指令、warp/wave 宽度、tile/stage、同步 |
| 本轮 INT16 partial 交接 | 满足证明条件时可无损缩窄存储 | 不保证加速；A100 本次配置已明确负向 |

对5090不能套用A100的原生INT4峰值：本项目CUDA12.8 legacy U4/S4路径在SM120受测SASS为INT8 IMMA加位操作，须另行审计与计量；本轮没有改动5090。AMD等平台可复用数学、guard和测量契约，但 CUDA/CuTe 的执行与加载布局需重写，尚无跨平台实测结论。

更大的 scale 动态范围、更长的 K 或不同码本也可能让整数路径大量回退，因此数学方法可移植不等于快路径覆盖率和速度比例可以照搬。

原始记录与复现入口：[v122证据索引](evidence/a100_o378_roof_v122/README.md)。

# O7：权重转换阶段缓存系数（v132，独立候选）

## 为什么不同于已失败的 v76

当前 v78 每组对输出做 `coefficient = A_factor * W_factor`，再以该系数加权 INT32 partial。
O7 的 A factor 为2的幂。v76 曾由每个 CTA 在 GEMM 内临时生成10行 shared 系数表，增加标量读写/local，四样本配对下降约24%；该实现不重做。

本轮把 `W_factor * 2^d` 的生成移到**可缓存的权重转换阶段**，每份权重只生成一次，GEMM 用原 cp.async pipeline 读取缓存表。
不是免费离线准备：若进入性能验证，建表必须计入 `weight_conversion`、`conversion_only` 与 Cold；compute-only/steady-state 才能缓存。
只选择固定10行；不扫表大小/布局，也不修改 O3/O8/5090/生产默认。

保持量化、G128 scale、两路原生 INT4、全 K INT32 顺序与最终 FP32 不变。
原始 32×N factors 留在扩展 buffer 的前部；整数 guard 和 FP32 fallback 保留。
GEMM 入口仍在 GPU 检查全部 Af 是否为正2次幂且≤512，不满足则调用旧 v78 整数 body；这段检查仍计入 GEMM。

## 准确的代价与投入门槛

布局为 `[32 groups, N/128, 16 N8片段, 10 exponent, 8 columns]`，来自现有 CuTe 输出坐标，保持相邻两列连续。
实际选中的 coefficient 由原 INT32 guard 保证不溢出；未选中表项采用定义明确的无符号移位，不用截断后的值冒充合法系数。

| 项目 | v78 | v132 |
|---|---:|---:|
| 每 CTA shared | 34304 B | 43520 B |
| 每 G128 payload＋metadata 读取 | 17152 B | 21760 B（+26.87%） |
| 4096×4096 额外权重缓存 | 0 | 5242880 B（5 MiB） |
| CTA / warp / stage | 64×128×128 / 4 / 2 | 保持 |

预先固定：旧控制完整SASS不变；同entry仍32+32原生INT4、16 LDSM、12 async copy、1热barrier；循环内无建表STS。
IMAD族至少减少20%、总静态指令最多增加2%、allocated≤168、热local≤1，实际驻留至少3CTA。
门槛通过后才考虑表项GPU正确性、安全检查与24样本三轮交错配对（warmup1000/repeats200），不做小规模性能筛选。
这些门槛是投入筛查，不是“少指令就一定更快”：额外供数、动态索引和shared访问可能抵消算术节省。

首次提交时，本地检查既有 v105 的24份O7 metadata：3,145,728个Af全部可表示，最大512，1536/1536个M tile满足查表范围。
这只是Af侧范围，不代替完整原INT32系数/乘积/前缀guard，不是新GPU性能或MSE结果。
源码、table地址双射/连续性、整数系数和范围单元测试通过；A100编译结果如下。

实现：[probe_o7_cached_coeff_codegen.py](../scripts/probe_o7_cached_coeff_codegen.py)；CUDA builder 与独立 entry：[roof_o7_cached_coeff_probe.cu](../csrc/sm80/roof_o7_cached_coeff_probe.cu)。

## A100 编译结果：查表成本抵消算术节省，停止

本地源码提交并推送 `81f6412e8ec2dc15e99917deaa5f50377400a547` 后，A100 在项目目录 fetch/ff-only 同步；CUDA 12.8.93 与 CUTLASS commit 保持固定。
原 v78 控制的完整机器码与旧证据一致，候选没有进入正式扩展。

| 同一整数热循环的静态指标 | 原 v78 | v132 |
|---|---:|---:|
| 指令 / G128 | 383 | 564（+47.26%） |
| IMAD 族，包含寻址等 | 158 | 224（+41.77%） |
| LDS 族指令 | 15 | 71 |
| LEA 族指令 | 4 | 46 |
| Allocated registers / thread | 168 | 168 |
| 热循环活跃寄存器峰值 | 166 | 165 |
| 热 local load / store | 0 / 0 | 0 / 0 |
| 原生 S4×S4 / U4×S4 MMA | 32 / 32 | 32 / 32 |
| LDSM / async copy / CTA barrier | 16 / 10 / 1 | 16 / 12 / 1 |

原路径复用少量行/列 factor，在寄存器内相乘；新路径则按输出坐标和行指数动态寻址 coefficient 表。
虽然不再于 GEMM 内生成整张表、没有热 STS，也没有 spill，更多 shared 查表与地址构造仍增加了真实指令工作。
因此，**减少源代码中的乘法，不等于减少机器指令或整数管线压力**；不能把全部 IMAD 都归因于 scale 乘法。
ptxas 对整个新 entry 报告 0 B stack / spill stores / spill loads，资源没有改善，额外5 MiB权重缓存和26.87%供数增量也仍存在。

原工作量上限1.02倍、IMAD至少减少20%两个门槛失败，其他编译结构门槛通过。
额外入口范围检查和权重建表成本还未包含在上述热循环对比中。
**停止该缓存表候选，不做相邻表大小/布局扫描，不启动 GPU 性能测试。**
静态多47.26%不是实测慢47.26%；没有新的Event、MSE、CV、NCU、GPU安全性或实际驻留结论，建表kernel也未执行。
当前最佳仍为 O3 v89、O7/O8 v78；已确认的转换候选保持，正式默认/扩展与5090不变。

11份原始文本、生成源码及SHA见[冻结证据](evidence/a100_o378_roof_v132/README.md)。
可重放源码、24样本Af范围、表地址与整数数学、原始证据/失败门槛；这些CPU检查不是GPU正确性验收。

## 对瓶颈与跨平台迁移的启示

当前 GEMM 已经消除了逐组 FP32 转换/累加；主要问题仍是原生 MMA 依赖、整数加权、地址工作和供数同步的组合。
本轮没有改变 NCU 结论，也不支持将瓶颈改称 HBM 带宽不足。
只有减少整体工作，或在不明显增加供数/资源压力的情况下隐藏依赖，才值得进入完整24样本测试。

将静态工作移到可缓存权重转换阶段是可移植原则，但本次具体查表实现没有已确认收益。
跨平台需重新比较缓存表读取/寻址与直接计算成本，保留Cold计费、溢出guard和正确回退；不能只迁移缓存思路就假定加速。
其他可迁移策略与5090限制见[瓶颈与平台迁移说明](o3_o7_o8_bottleneck_portability_20261007.md)。

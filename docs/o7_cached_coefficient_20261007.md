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
源码、table地址双射/连续性、整数系数和范围单元测试通过；A100编译待执行。

实现：[probe_o7_cached_coeff_codegen.py](../scripts/probe_o7_cached_coeff_codegen.py)；CUDA builder 与独立 entry：[roof_o7_cached_coeff_probe.cu](../csrc/sm80/roof_o7_cached_coeff_probe.cu)。

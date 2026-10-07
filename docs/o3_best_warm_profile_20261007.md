# 当前最佳 O3：预热后 NCU 瓶颈核对（v136）

## 结论

**保持两路原生 INT4，当前最佳和正式默认均不变。** 本轮补齐当前最佳 O3 的预热后诊断，
没有新增优化候选、24样本 Event 性能或加速成绩。O7/O8、5090及量化数据未改动。

主要限制仍是 **MMA 依赖与数学管线发射压力，加上有限的可运行 warp 和供数等待**，
不是逐组 FP32 转换/FMA，也没有证据支持把 HBM 带宽当首要瓶颈。
这次采集排除了仅依据旧版清缓存回放判断优化方向的问题，但不证明已经达到不可突破的性能上限。

## 采集对象与结果

A100，`layer_12_o_proj`，4096³，原 v89 `adangel_roof_o3_grouped_cta_candidate`；
NCU `--set full --replay-mode application --cache-control none --clock-control none`。
每个进程预热1000次后采一个正式入口，共50次应用回放，全部使用相同输入及冻结 GEMM cubin。
只构建既有 host/转换适配器，没有重编译 GEMM 或正式扩展。

| 指标 | 本轮观察 |
|---|---:|
| NCU Duration | 0.376160 ms |
| 本次 GPC 时钟 | 1.288170 GHz |
| Tensor 管线活跃率，active 分母 | 65.28% |
| Issue active | 42.95% |
| Eligible warps / scheduler | 0.6485 |
| 寄存器 / 每SM最大CTA / 实际occupancy | 168 / 3 / 18.01% |
| L2 命中率 / DRAM 吞吐率 | 94.41% / 17.87% |
| LDS、LDSM excessive shared wavefronts | 均为0 |
| 全入口动态 warp 指令 | 88.231936M |
| INT4 MMA / 最终 I2F / FFMA 动态指令 | 16.777216M / 0.524288M / 0 |
| 此样本 MSE vs O0 | 0.0003752320504872409 |
| 此样本 MSE vs 原最佳输出 | 0，逐位一致 |

50份回放 receipt 均验证 finite FP32、逐位输出、输入SHA、实际资源与二进制。
所有2048 CTA均走原安全整数路径；SASS保留 S4×S4 与 U4×S4，不含 INT8 Tensor Core。
存在每线程16B local 分配，全入口各8192条LDL/STL是非热循环工作；**不能宣称整个kernel零local**。

NCU时长不是正式 Event 的24样本中位数，不能将0.376160与上一轮0.435712相除声称加速。
本次的1000次预热与O8旧诊断的50次预热也不同，不能把两份NCU当成同进程配对性能测试。

## 等待落在哪里

按真实寄存器数据流核对全部2200条入口解码指令，将323条整数热循环指令标注到消费者阶段；
动态计数与10953个未发射采样全部闭合。

| 消费者阶段 | Wait采样 | Math-pipe throttle采样 | Short-scoreboard采样 |
|---|---:|---:|---:|
| high MMA：第一段K64 | 596 | 1046 | 163 |
| high MMA：第二段K64 | 776 | 673 | 333 |
| high partial ×16 | 348 | 51 | 0 |
| low MMA：第一段K64 | 280 | 365 | 0 |
| low MMA：第二段K64 | 628 | 521 | 46 |
| partial × W_factor + 全K整数累加 | 370 | 26 | 208 |

MMA消费者占 **53.79%的Wait、94.56%的Math** 采样；重构与加权两步占16.94%的Wait。
加权消费者还有208个short-scoreboard采样，说明不能把scale路径只看作整数乘法，供数就绪也需要考虑。
这些是消费者位置的采样，不是各阶段耗时，更不是可直接消除的延迟比例。
Issue active未满且eligible<1，才使依赖隐藏值得关注。[NVIDIA NCU说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons-not-issued)

乘16与逐组加权合计33.554432M动态warp指令，占全入口38.03%；这是工作量，不是38.03%的时间。
1285个barrier样本落在循环控制消费者，不能把它们算成分支本身的开销。
LDS/LDSM没有excessive，但异步copy仍有171744个excessive wavefront，不能将整条shared路径称为零额外工作。

## 对下一步的约束

当前每线程64个最终accumulator、32个partial及A/B fragment挤占寄存器，实际只能3 CTA/SM。
增加链或warp虽然可能隐藏等待，但已有完整配对证实其寄存器、供数和额外工作可能抵消收益。

| 方向 | 当前判断 |
|---|---|
| 减少逐组FP32转换/FMA | 已完成，不再作为新的优化目标。 |
| high/low拆成更多独立链 | v125、v135完整24样本分别−5.71%、−6.48%；不重测相近变体。 |
| 增加warp、缩短fragment生命周期 | v98/v104、v94已有负向/资源证据，不只为了occupancy改配置。 |
| 下一组fragment预加载 | v128/v134的地址与local开销未过门槛，不重复相同iterator。 |
| HBM或shared bank-conflict优先 | 当前94.41% L2命中、17.87% DRAM吞吐及读取excessive=0，不支持其作为首选。 |
| 后续值得实现的候选 | 必须在保持两路INT4、原scale和guard的同时，实质改善MMA/重构/供数交叠，并说明为何不会重现上述代价。 |

本轮没有找到满足最后一项、且不同于已测机制的新候选，因此不为增加轮数进行参数扫描。
当前最佳仍是 O3 v89、O7/O8 v78。约0.220347ms只是1410MHz下固定工作、理想重叠的MMA容量必要下界，
不是当前实现保证可达的速度；各资源下界不能相加，也不能将与实测的差全部视为可优化开销。

## 复核

[原始证据及SHA](evidence/a100_o378_roof_v136/README.md)保存50份receipt、原始CSV、PC角色、命令和完整回放日志。
完整 `.ncu-rep` 另保存在同名归档中；没有删除离群采样、改变GPU时钟或等待GPU空闲。
此次为诊断及既有输出回归，**不是新的24样本性能验收或内存安全测试**。

```bash
python -m pytest tests/unit/test_o3_best_warm_ncu.py \
  tests/unit/test_o3_mma_phase_stalls.py tests/unit/test_roof_v136_evidence.py -q
# 只重分析既有采集，不再次运行GPU：
python scripts/run_o3_best_warm_ncu.py \
  --output reports/o378_roof_v136_o3_warm_ncu --analyze-only
```

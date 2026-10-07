# O3：高低路分别加权的依赖消除候选

日期2026-10-07，v125。**完整24样本配对确认吞吐下降5.71%，不采用；原始输出/MSE不变。**

当前v89每条链是high MMA→high MMA→×16→low MMA→low MMA→乘factor并累加。本轮保持32个partial寄存器，将其分为16个low和16个high；每个M片段有4+4条独立、长度为2的MMA链。随后分别更新同一full-K accumulator：

```text
acc += low_dot  × factor
acc += high_dot × (16 × factor)
```

×16作用在可复用的列factor上，不再插入两类MMA之间。原quantizer、G128、三stage、grouped CTA64×128×128、4warp、原始输入/metadata/转换、guard/fallback以及FP32 epilogue均保持。没有新的scale表示或转换误差。

拆开的项可能各自超出INT32，即使最终和在范围内。因此两次更新明确使用PTX `mad.lo.s32` 的低32位语义，高factor通过无符号×16生成；模2³²代数保证合并后等于原先被guard证明可表示的整数prefix。禁止依赖C++有符号溢出或加入饱和。CPU含抵消/边界/随机代数检查，GPU仍须另行验收。

该指令取乘积低位后与32位C相加；不是饱和乘加，依据[NVIDIA CUDA12.8 PTX mad定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad)。

查重：v67先重构`low+16high`再乘factor；v78/v79用high作为low MMA的C；v96/v97只尝试跨N64交错；v115使用INT16 packing/DP2A。本轮没有这些重构或packing，直接用两个低32位乘加更新同一整数accumulator。不是“把high移到循环外”——每G128的两路乘积和各自scale仍完全保留。

预先固定编译投入门槛：同entry两条原生INT4、32+32 MMA，16 LDSM、9 async copy、1 CTA barrier，旧控制完整编码一致；allocated≤168、热local=0；机器码确有32条同符号两MMA链且静态未完成链峰值≥8，循环静态指令增幅≤8%。此门槛检查是否值得测试，不预测加速。

门槛通过才查实际资源、GPU数值/guard/安全性，并直接做24样本×3轮、warmup1000/repeats200配对，不做小规模性能初筛。确认GEMM收益后再补四模式。失败即停止，不扫相邻链/排布、不迁移O7/O8、不修改5090。

## 编译后的独立资源复核（在任何候选GPU执行之前决定）

原始零spill门槛结果**保留为失败，不回写为通过**：323→337条（+4.33%），168 allocated/163 peak GPR，4条热LDL、0条热STL；原生MMA64/LDSM16/copy9/barrier1均不变。机器码确有32条同符号、长度2的MMA链，静态未完成链峰值8；旧控制完整编码相同。

用户此前明确允许少量spill，只要正确且更快。此次目标依赖结构已真实改变，因此在任何候选GPU执行、计时或MSE之前，单独决定做一次完整配对取舍：允许已审计的4条local读取、不允许local写，其他原门槛保持；实际资源API必须确认仍≥3 CTA/SM后才准许运行。**这不是宣称spill无害，也不修改未来通用验收门槛。** 初始失败和本次复核均保存，性能不好就停止，不扫描相邻版本。

运行入口为`benchmark_o3_split_weighted.py`，输出额外记录初始gate及复核依据。原控制、guard、输入、转换、计时协议不变。

## A100完整结果：停止，不迁移O7/O8

源码先在本地提交并推GitHub，再在A100项目内fetch/ff-only同步。编译commit为`99a13828f92d7d8cbca2c4be795295cf8c88f41a`，运行commit为`d7ecb5727bd7c11e56df16124dd9e90930ac793b`；CUDA12.8和原pinned CUTLASS不变。

24真实样本×3轮、warmup1000/repeats200、同输入/进程/stream、循环交错顺序；无锁频、不等GPU空闲、不删除CV失败记录。每条记录200次正式测量，合计144条记录、28,800次计时执行；GEMM/total字段不是两套独立执行。

| 指标 | 当前最佳O3 v89 | 新v125 |
|---|---:|---:|
| Compute-only median ms | 0.436736 | 0.465920 |
| 配对吞吐比（旧时间/新时间） | 1.000000 | 0.942857 |
| 配对比值bootstrap 95%描述性区间 | — | [0.940397, 0.944079] |
| CV≥3%记录 | 1/72 | 1/72 |
| MSE vs O0 median | 0.006653010287410 | 相同 |
| MSE vs O0 mean | 0.007578847013303 | 相同 |
| CUDA查询寄存器/线程；CTA/SM | 168；3 | 168；3 |
| CUDA local size bytes/线程 | 16 | 24 |

配对吞吐**−5.71%**按每样本三轮配对比值的中位数汇总，并非表内两个总体中位数直接相除。区间描述本次固定trace配对，不是跨模型/平台置信保证。所有144条输出与旧实现逐位相同、新旧输出MSE=0，payload/scale/status相同；相对O0的MSE并未因模整数计算变化。

96项小M/N、完整K4096合成检查、8项非法输入拒绝、2项grouped/tail坐标检查通过。仅对候选entry进行的有限memcheck/synccheck均0 errors，不外推为4096³ sanitizer或racecheck全覆盖。没有新增NCU、正式转换/Cold/steady性能测试；转换未改，但不能把GEMM差值当作直接测得的端到端差值。

## 瓶颈insight：短链不等于更高并行吞吐

| 同一G128热循环 | 旧 | 新 |
|---|---:|---:|
| MMA链数量×每链长度 | 16×4 | 32×2 |
| 静态未完成链峰值（非硬件同时在飞数） | 8 | 8 |
| 原生INT4 MMA / LDSM / async copy / barrier | 64 / 16 / 9 / 1 | 不变 |
| 普通IMAD | 65 | 129 |
| IMAD族合计 | 91 | 133 |
| 热local load / store | 0 / 0 | 4 / 0 |
| 热循环静态指令 | 323 | 337（+4.33%） |

每线程partial槽位仍32个。旧链在一个fragment中合并high/low；新版为high/low各保留一套，因此同一时刻覆盖的独立输出减少，MMA总工作和链容量并没有增加。与此同时，原来的每输出一次factor加权变为两次，IMAD族工作增加46.15%，并出现4条local读取。缩短单条链所获得的收益不足以抵消整体成本，实际反而更慢。

上述是代数、SASS与完整配对共同支持的解释；**未用本轮NCU隔离因果，不能声称5.71%全部由spill或某一个stall造成。** 也不能由此推断所有短链方案必然失败。

停止该路线，不扫相邻分组/布局、不迁移O7/O8。当前最佳GEMM仍为O3 v89、O7/O8 v78；已确认的转换候选也不回退。正式默认/扩展及5090代码保持，主吞吐目标尚未达到。原始编译、Event、MSE及有限安全日志见[v125证据](evidence/a100_o378_roof_v125/README.md)。

```bash
python -m pytest tests/unit/test_split_weighted_codegen.py -q
python scripts/probe_split_weighted_codegen.py --output reports/o378_roof_v125_codegen
```

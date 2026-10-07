# O8：packed FP6 转换；当前 GEMM 瓶颈与平台迁移

日期：2026-10-07。本轮仅新增一个独立转换候选，**不改变正式默认、原生扩展、O3/O7、5090 后端或源量化**。GEMM 仍用当前 v78。未重复扫描 CTA、warp、stage、累加链或 cache 参数。

## 为什么选择这个方向

当前最佳已使用寄存器 partial、八条合并 MMA 链和带安全 guard 的全 K 整数累加。不能再把瓶颈说成“每 G128 都做一次 FP32 FMA”。目前剩余的是两路 INT4 MMA、整数 factor 加权、寄存器依赖以及供数/同步的组合开销。

已有 O8 预热 NCU 的 eligible warps/scheduler 为 0.7464，issue active 46.90%；Tensor/FMA pipeline 分别58.95%/37.96%。未发射 PC 样本中 wait/math/barrier/MIO 为35.54%/29.11%/11.74%/9.29%，**不是耗时占比**。L2 hit96.97%、实际 DRAM读约19.87MB，不支持把 HBM 带宽当首要瓶颈；LDS/LDSM excessive wavefront均为0，也不能笼统归因 shared bank conflict。详见[当前瓶颈和原始证据](o3_o7_o8_bottleneck_portability_20261007.md)。

GEMM 的多项低成本邻近方案已经查重：v43/v76/v87 scale复用/查表/移位，v84/v95 pipeline，v98/v104增加warp，v111扩大合作CTA，v115 DP2A后处理，v119取消MMA volatile，v121只读factor，以及v122跨warp partial交接。没有新的机制证据时，不重复这些路线。

因此本轮选择**辅助的 conversion/端到端优化**：O8 FP6激活原来逐元素解码、packing和计算平方和，新实现直接处理每个32-bit word中的四个编码。它不提高纯MMA peak或GEMM吞吐，不能用转换收益冒充主要GEMM目标已完成。

## 实现与正确性

保持 E2M3→Q6、F=2、RNE和原两级scale。这里是项目已约定的实验变体“NVFP6”（E2M3＋E4M3/FP32 scale），不是宣称其为标准硬件格式。每个源FP6编码原本就占一个byte，没有重新压缩源格式。四个byte并行计算指数分支和RNE幅值，再逐byte恢复二进制补码；中间值不超过128，不会向相邻byte进位，负零仍为0。

得到的四个有符号INT8整数在一个寄存器中，用 `dp4a.s32.s32(q,q,sum)` 计算精确平方和，再分别压入低/高INT4 payload。DP4A是普通整数dot指令，**不是把GEMM换成INT8 Tensor Core**；GEMM仍加载同一份v78 CUBIN、调用同一个函数。

每元素绝对值≤30，单个dot平方和≤3600，整G128≤115200。后续原v73 subwarp归约、row metadata、factor/anchor、饱和范数和CTA guard完整保留；不放宽guard，不省略统计，不修改量化或MSE参考。

[NVIDIA PTX的DP4A定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a)规定了四个packed byte乘积的INT32累加。本轮必须同时验证GPU实际指令和数值，不能只凭PTX名称判定性能。

与已测方案的区别：v66只是单个FP6元素的无分支改写；v53是向量内存访问；v73是转换与metadata融合。此次保留后两者，改变寄存器内数据组织，并用一次dot替代四次标量平方计算。也不是重测v118的NVFP4 nibble布尔解码/POPC。

## 编译及资源审计

| 指标 | 原v73 O8激活转换 | 新候选 |
|---|---:|---:|
| 全entry静态指令 | 696 | 408（−41.38%） |
| 寄存器/线程 | 23 | 29 |
| CUDA查询的CTA/SM，256线程/CTA | 8 | 8 |
| shared bytes | 256 | 256 |
| stack/local/spill | 0 | 0 |
| CTA barrier | 1 | 1 |
| 原生 `IDP.4A.S8.S8` | 0 | 4 |

所有旧库entry的完整SASS编码保持不变。这里的41.38%是**静态指令减少**，不是实测提速。

保留门槛变更记录：首次保守门槛要求寄存器不增加，因23→29未通过，未执行候选GPU。随后在任何数值/性能运行前，单独用CUDA资源API核对：两者都是8 CTA/SM。第二次资源审查要求寄存器≤32、实际驻留不下降、零local、相同shared/barrier、至少5%指令减少和四条原生DP4A，均通过。不是根据性能结果事后放宽门槛，也未扫描寄存器上限；两次编译记录均保留。

首次源码commit `ecdccf88762580fcac6b668012c59ae66a81266b`；资源审查/正式编译 `9a30fbac703037299b767696dab610834332dc60`；运行程序 `cb568806c2e6d8faeac7fe89629bffff7217fb8d`。均先在本地实现、推GitHub，再在A100项目内fetch/ff-only。

## 测试口径

CPU覆盖全部256个byte编码（包括被忽略的上两位）、131072个两byte重复/互补组合和16384个随机word，检查逐bytecarry与低/高nibble packing。GPU再次比较scalar/SWAR/CPU的131072个word、精确平方和，以及合成G4096的四种计时模式和guard边界；数值预检已通过。

性能直接采用24个真实样本、三轮、warmup1000/repeats200、conversion-inner100、单stream、预分配、交错顺序。**没有小规模性能初筛**。输入源格式与v99 provenance逐项核对；两组绑定同一GEMM函数；每次Event计时结束后检查payload、scale、平方和、metadata、FP32输出和相对O6的MSE。不锁频、不等待GPU空闲、不丢弃离群值。

转换阶段仍批量摊销；Cold/steady total仍单次直接计时，不能相加推算。

## 完整24样本结果

共576条记录、345600个原始Event值，未过滤。表中ms为各样本三轮median再跨24样本取median；配对吞吐按同样本同轮旧/新延迟比汇总，不能直接用表中两列median相除替代。

| O8指标 | 原v73 ms | 新v123 ms | 配对吞吐提升 | speedup 95% CI | CV≥3%旧/新（各72条） |
|---|---:|---:|---:|---|---:|
| Conversion-only A | 0.057508 | 0.040589 | **+41.39%** | 1.410696–1.417567 | 0/0 |
| Conversion-only total | 0.091290 | 0.074414 | **+22.61%** | 1.224257–1.227342 | 0/0 |
| Compute-only GEMM | 0.481280 | 0.481280 | 0.00%（同一kernel） | 1.000000–1.000000 | 3/0 |
| Cold total | 0.580608 | 0.566272 | **+2.63%** | 1.025316–1.027174 | 0/5 |
| Steady-state total | 0.545792 | 0.529408 | **+3.04%** | 1.029014–1.035417 | 0/3 |

转换A的配对延迟下降约29.28%；总转换约18.44%；Cold和steady约2.56%/2.96%。这与表中吞吐提升是不同口径。W转换未改变，conversion-only W仍为0.033772ms、配对收益0%。

完整原始值和全部13项stage统计保留。Cold内的GEMM stage从0.468992变为0.474112ms，配对−1.08%；steady内也约−1.07%。其实际CUfunction完全相同，不能解释为GEMM源码被改慢；准备路径改变会改变后续执行状态，且本次未锁频，采样SM clock1290–1410MHz。没有对该stage做独立因果profiling，不能武断归因于某个cache或频率原因。直接total仍确认正向，收益小于转换阶段相对提升也与GEMM占主导相符。

Cold/steady候选分别5/72、3/72条total记录CV≥3%，全部保留，不宣称所有阶段严格低于3%。未以重测挑出更好的结果；正向结论仅适用于这次完整配对分布，CI为24样本重采样的描述性区间。

### 数值与安全

| 验收项 | 结果 |
|---|---|
| 24份O8源格式identity | 与v99逐项相同 |
| 576条新旧输出、payload/scale/norm/metadata | 逐位一致；finite FP32 |
| O8相对O6的MSE median / mean | **0.004411084910985704 / 0.004381379299073540**，未改变 |
| GPU decoder检查 | 131072 word的scalar/SWAR/CPU编码与平方和一致 |
| 合成正确性/guard | 32项四模式＋3项边界通过 |
| 有限memcheck / synccheck | 均0 errors |

sanitizer仅覆盖两个新增转换入口、有限合成小M/N、K4096；不是重新对完整4096³ GEMM进行全量race/memory/sync验收。GEMM二进制未改，正式扩展SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

**决定：保留v123为O8独立转换候选，当前GEMM最佳不变，不切换正式默认。** 已获得端到端的有限收益，但尚未突破主要GEMM瓶颈，不声称已达到有效吞吐上界。本轮不继续枚举相邻位运算、查表或寄存器上限变体，也不重新运行已否定的GEMM方案。

[冻结证据与重放说明](evidence/a100_o378_roof_v123/README.md)包含两次编译审查、原始Event、数值与安全检查日志；[文件索引](evidence/a100_o378_roof_v123/index.json)记录各文件SHA-256。可在CPU上重算统计，不需要重新运行GPU实验。

## 哪些可以移植

| 层次 | 可以复用 | 需要重新实现/验收 |
|---|---|---|
| 数学 | 源格式解码、精确平方和、dyadic scale分解、溢出guard和回退 | 不同格式、K、取值范围的证明 |
| 数据流 | 转换融合、静态权重缓存、寄存器partial、最终一次写回 | packing、MMA fragment坐标、shared布局 |
| 调度 | 独立MMA链、供数与计算重叠 | CTA/warp/stage、寄存器预算、同步API、真实SASS |
| 本轮packed转换 | 普通无符号位运算和字节独立算法 | 目标平台的dot指令或等价实现、实际occupancy和端到端收益 |

A100用 `cp.async`，不能将它描述成TMA；5090原后端的TMA/warp-specialization可以保留其设计原则，但不能直接复制A100的最佳参数。尤其本项目CUDA12.8受测的SM120 legacy U4/S4路径实际为INT8 IMMA加位操作，不能移植A100“两路原生INT4 peak/2”的数值结论。其他CUDA GPU或AMD等平台同样需要新的指令/数值/性能验收；本轮没有跨平台实测。

## 重放命令（均为独立候选）

```bash
python -m pytest tests/unit/test_nv6_swar_codegen.py -q
python scripts/probe_nv6_swar_codegen.py --output reports/fp6_packed_recheck
python scripts/benchmark_o8_nv6_swar.py --gpu-build reports/fp6_packed_recheck \
  --validate-only --output runs/fp6_packed_validation_recheck
python scripts/benchmark_o8_nv6_swar.py --gpu-build reports/fp6_packed_recheck \
  --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100 \
  --output runs/fp6_packed_full24_recheck
python scripts/analyze_nv6_swar.py --input runs/fp6_packed_full24_recheck \
  --output runs/fp6_packed_full24_recheck/analysis.json
```

所有输出目录必须为新的项目内目录。无须重新采集trace或重编译正式扩展。

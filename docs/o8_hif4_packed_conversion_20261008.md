# O8 多级 scale 权重转换优化（v138）

## 范围与当前状态

本轮仅优化 HiF4→Q4 权重转换；两侧均使用当前 v123 FP6 激活转换与 v78 两路原生 INT4 GEMM。
没有更换量化、G128 scale、整数范围guard、正式默认或5090代码。
编译、正确性和24样本配对测试完成。权重转换配对吞吐提升83.17%，转换总吞吐提升26.25%，
Cold吞吐提升2.98%；GEMM和steady-state无新增收益。保留为当前O8转换最佳独立候选，正式默认不变。
Cold内W分项仍有较多CV超标，不能宣称全部计时分项通过严格稳定性门槛。

## 本轮实测结果

对照与候选都用同一v123激活转换及v78 GEMM，只切换HiF4权重转换。
下表延迟先取各样本三轮中位数，再取24样本中位数；百分比来自逐样本配对速度比，
不是用表中两列中位数直接相除。未与跨轮历史绝对延迟拼接计算累计收益。

| 指标 | 原HiF4转换 ms | packed HiF4 ms | 配对吞吐变化 | 配对延迟减少 |
|---|---:|---:|---:|---:|
| Conversion-only：W | 0.033802 | 0.018540 | +83.17% | 45.41% |
| Conversion-only：A | 0.040535 | 0.040499 | +0.10%，实现未改 | 0.10% |
| Conversion-only total | 0.074440 | 0.059110 | +26.25% | 20.79% |
| Compute-only GEMM | 0.481280 | 0.481280 | 0.00% | 0.00% |
| Cold total | 0.566272 | 0.549888 | +2.98% | 2.89% |
| Steady-state total | 0.529408 | 0.529408 | 0.00% | 0.00% |

W、转换total、Cold total配对速度比的bootstrap 95%区间分别为
`[1.82431,1.83681]`、`[1.25936,1.26709]`、`[1.02980,1.03059]`。
steady-state缓存权重转换，因此本轮不应给它带来算法收益；GEMM函数和机器码完全相同。
Cold中单次GEMM分项点估计反而慢0.22%，不能将转换减少包装成GEMM优化。

| 输出精度 | 原HiF4转换 | packed HiF4 |
|---|---:|---:|
| MSE vs O6：median | 0.004411084910985704 | 0.004411084910985704 |
| MSE vs O6：mean | 0.004381379299073540 | 0.004381379299073540 |
| 新旧输出之间的MSE | 0 | 0，逐位一致 |

G128 scale、payload、精确平方和、factor/anchor、guard及输出在所有配对记录中相同。
`layer_24_o_proj`两侧均有12/2048 CTA走既有FP32缩放回退；其余23样本全走整数路径，
与v123既有证据一致，不是本轮新回退。两种路径都仍使用原生INT4点积。

### 稳定性及中断处理

24样本×3轮，每侧每种模式72条记录；576条完整记录、345600个原始Event值全部保留。
conversion-only W和total双方均0/72 CV≥3%；compute-only为旧3/72、新0/72；
Cold total为旧3/72、新2/72；steady total为旧1/72、新0/72。

**Cold内独立摊销的W转换分项为旧0/72、新54/72 CV≥3%**，不是全部稳定通过。
该分项在直接端到端测量之后另行预热并批量测量；若干序列前段逐步下降，而非仅一个孤立尖峰。
这与负载切换后的时钟/热状态过渡相容，但未连续采样时钟，不能确定归因于调度或频率。
抽样SM clock为1275–1410MHz。保留原始值及限制，不删除前段、不改计时器掩盖CV。
因此W收益主要依据稳定的conversion-only轨，Cold收益依据直接total轨，不把Cold-W的CV问题忽略。

SSH在完成20样本后断开，进程终止。原483条记录原样保留；保留前480条完整记录，
最后一个不完整样本的3条记录不并入最终配对，整组补测后4样本（96条）。
续测保留原样本索引/交错顺序，重新验证全部数据SHA、数值和二进制；不按快慢或CV选择。
最终合并结果及两次采集的commit/SHA见`recovery.json`，不是另一次候选迭代。

## 借鉴与实际改动

HiFloat4官方GPU代码提供分层量化/反量化参考，而非可直接用于本实验的HiF4 GEMM。
Marlin提供打包解码、字段复用和常量修正融合的参考。固定源码与差异见
[开源实现研究](scaled_kernel_reference_study.md#本次补查hifloat4-与新版-marlin-的多级-scale)。

原转换对每个元素计算：

```text
q = sign × RNE(magnitude × 2^(micro8 + micro4) / 4)
```

新转换在一个32-bit word中同时处理8个nibble；micro8覆盖8元素、micro4分别覆盖前后4元素。
针对局部指数和为0/1/2，分别精确计算 `RNE(m/4)`、`RNE(m/2)`、`m` 并按字段选择。
负零归零，舍入仍为nearest/ties-to-even，不出现跨nibble进位。
平方和用标量DP4A精确计算，继续用于原有溢出guard；**DP4A不是INT8 Tensor Core MMA**。
G128外层E6M2 scale、factor/anchor生成、最终scale恢复与原实现一致。

| 转换入口资源 | 原HiF4转换 | packed HiF4 |
|---|---:|---:|
| 静态SASS指令数（整个入口） | 544 | 312 |
| 寄存器/线程 | 31 | 22 |
| local/spill | 0 | 0 |
| shared/CTA | 256 B | 256 B |
| 资源允许CTA/SM | 8 | 8 |

这是转换工作量的变化；MMA数量、GEMM资源和GEMM SASS不变。

## 补充：两路INT4的数据拆分与存放

对有符号8-bit激活整数，取低四位为UINT4、高四位为INT4：

```text
u = uint8(a)
low  = u & 15                 // 0…15
high = (u >> 4); high -= (high >= 8 ? 16 : 0)  // −8…7
a = low + 16 × high

a = −37: 1101 1011 → high=−3, low=11 → −3×16+11=−37
```

拆分是精确整数恒等式，不增加量化误差。Q4权重只有一份，两路MMA复用同一个权重fragment。
O7的Q8激活直接拆分；O8的Q6整数先符号扩展为INT8后拆分，仍完全精确。
最佳候选先算high partial、乘16、再将low MMA累加到该INT32 partial，而非物化两张完整输出矩阵。

HBM中的逻辑布局（M=N=K=4096，G128）如下；每字节的低/高nibble分别存放相邻偶/奇K元素：

| 数据 | 紧凑物理buffer的逻辑视图 | payload字节数 |
|---|---|---:|
| 激活low/high两平面 | `uint8[2,32,4096,64]` | 16 MiB，等于原INT8矩阵 |
| 一份Q4权重 | `uint8[32,4096,64]` | 8 MiB |
| scale/factor/anchor/guard | 独立buffer，非payload nibble的一部分 | 另计 |

`32`是G128组数，`64`是128个四位数的打包字节数。激活先放完整low平面，再放完整high平面；
每个平面内按group→row→packed K组织，不是逐元素交错low/high。
以字节计，low位置为 `(group*M+row)*64 + k_in_group/2`，high在其基础上加 `M*K/2`。

GEMM通过16-byte `cp.async`把tile拷入swizzled shared memory，之后由`ldmatrix`/CuTe copy送入MMA fragment。
寄存器仍是打包位：一个32-bit寄存器容纳8个四位payload；各线程持有的元素由CuTe fragment映射决定，
不能理解成每线程简单持有连续8个K。单个m16n8k64原子每lane用4个A寄存器、2个B寄存器，
产生4个INT32 accumulator寄存器。low按U4解释、high和weight按S4解释；
scale和partial各有自己的寄存器/缓冲区，不改变这两个payload平面的布局。

## 验证与复现

CPU证明覆盖1,048,576个打包word/metadata组合，以及随机word、负零、ties和共享字段索引。
GPU对应编码/平方和的1,048,576组合、32项合成四模式、3项guard边界及24样本MSE均通过。
新增转换及decoder的定向memcheck/synccheck在小M/N、K4096下均0错误；
不声称重新完成整个GEMM或完整4096³的sanitizer。GEMM仍为已审计的同一v78 cubin，
原生U4×S4/S4×S4两路保留；21个既有转换/测试入口机器码不变。
首次编译审计因匿名测试入口符号哈希改变而失败，严格限定该符号归一化后机器码对照通过；
首次失败记录保留，没有放宽机器码或资源门槛。
正式计时沿用双轨：转换阶段Event内重复100次摊销，GEMM和端到端单次直接计时；
24样本×3轮，预热1000、测量200，不锁频、不等待GPU空闲、不删除离群值。

```bash
python -m pytest tests/unit/test_hif4_swar_codegen.py -q
python scripts/probe_hif4_swar_codegen.py --output reports/hif4_packed_new
python scripts/benchmark_o8_hif4_swar.py --gpu-build reports/hif4_packed_new \
  --output runs/hif4_packed_full24_new
python scripts/analyze_hif4_swar.py --input runs/hif4_packed_full24_new \
  --output runs/hif4_packed_full24_new/analysis.json
```

所有输出目录须为新的项目内目录。独立编译候选库，不重编译或替换正式扩展。

[完整原始证据、SHA与回放测试](evidence/a100_o378_roof_v138/README.md)。
Tilus的布局/向量转换思想另见[源码对照](scaled_kernel_reference_study.md#tilus借鉴布局与打包转换不照搬-fp16-计算路径)；
本轮HiF4算法在阅读Tilus前已经实现，不把当前收益追溯归因于该论文。
这次进展改善conversion和Cold，并未缩小GEMM与MMA容量下界之间的距离；GEMM优化仍是核心目标。

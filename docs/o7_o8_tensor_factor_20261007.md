# O7/O8：Tensor Core 生成 scale 外积候选（v129）

## 为什么测试

当前最佳每 G128、每线程有64次 `Af[row]*Wf[col]` 系数乘法，再有64次 partial 加权累加。
v129只尝试把前一部分系数外积交给 U8×U8→S32 Tensor Core；主点积仍是两路原生 INT4。
不是将主 GEMM 改为 INT8，不改变 source quantizer、G128 scale、整数求和顺序或 FP32输出。
O3没有行列双 factor乘法，不直接套用此候选。

构造一个只有K=0非零的 `16×16` A系数矩阵和 `16×8` B系数矩阵，
一次 `m16n8k16.u8.u8.s32` 产生16×8个精确系数，输出坐标与原INT4 MMA相同。
每线程原64次标量系数乘法，理想上替换为16次warp级helper MMA；后续partial加权仍保留。
这是转移执行单元的假设，不是已经证明加速。Tensor服务、输入packing、依赖和寄存器也会增加。

依据固定CUTLASS的 `SM80_16x8x16_S32U8U8S32_TN` 及其 `MMA_Traits`，
用CuTe identity tensor验证所有输入/输出坐标，不猜lane布局。
[NVIDIA PTX 8.7的整数m16n8k16说明](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8)。

查重：v71/v87移位、v76共享系数查表、v105/v108相等/同质scale、v115 DP2A、v127面板预载均不是此机制。

## 正确性与投入门槛

- 原full-K系数/乘积/前缀guard保留；CTA开始时另外检查所有Af/Wf是否属于[0,255]。
- 超范围（包括负整数）完整返回原v78整数路径；原flag=1仍走原FP32回退；不截断、不重定标。
- 新range guard的读取/归约/同步计入GEMM及端到端，不冒充离线免费工作。
- CuTe host坐标与CPU穷举U8外积先通过；同entry审计必须区分64条原INT4与16条新增系数INT8。
- 编译投入门槛：allocated≤168、热local≤1、热循环静态指令≤旧版1.01倍、IMAD族减少≥20%；
  payload MMA、LDSM、copy、热barrier不变，原控制完整机器码一致。
- 通过后才核对完整24样本的双侧U8覆盖（不是仅Af），实际≥3CTA/SM、GPU正确性/MSE与安全，
  然后直接24样本×3轮1000/200配对；无小规模性能初筛。正向再补四模式。

初步只读检查48份既有Af快照：O7/O8最大Af为512/3584；按M64且覆盖全K，
Af单侧U8覆盖均值为92.45%/49.61%。这不是包含W、原guard或新开销后的覆盖，更不是性能成绩。

## A100 实际编译结果：停止，不进入 GPU 性能测试

源码 `182f79b29c3662475b1cfa1aab88ec8fbb191e79` 先推送 GitHub，再在 A100 项目目录 fetch/ff-only。
CUDA 12.8.93、固定 CUTLASS；原 v78 对照的完整 SASS 编码不变。
CuTe host 验证了 8192 个输出坐标及全部 A/B fragment 坐标；CPU 穷举了 65536 对 U8 系数。
这些验证不等于候选已在 GPU 上通过数值或安全测试。

| 同一整数热循环的编译指标 | 原 v78 | v129 系数外积路径 |
|---|---:|---:|
| 静态指令 / G128 | 383 | 461（+20.37%） |
| IMAD 族（含地址/移位等，不全是 scale） | 158 | 115（−27.22%） |
| 普通 IMAD | 128 | 69 |
| 原生 signed / unsigned INT4 MMA | 32 / 32 | 32 / 32 |
| 新增系数 U8×U8 MMA | 0 | 16 |
| PRMT 字节重排 | 0 | 54 |
| 热 local load / store | 0 / 0 | 3 / 3 |
| Allocated registers / thread | 168 | 168 |
| 静态活跃寄存器峰值 | 166 | 166 |
| LDSM / async copy / CTA barrier | 16 / 10 / 1 | 16 / 10 / 1 |

**系数乘法确实迁移到了 Tensor Core，但输入构造成本抵消了节省。**
同一正式候选 entry 仍有两路 `IMMA.16864.S4.S4` / `U4.S4`，
新增的 `IMMA.16816.U8.U8` 只计算系数，不是把 payload 换成 INT8。
CuTe 将稀疏系数填入 INT8 fragment 时产生了额外 packing、选择、读取和控制指令，
例如 PRMT 0→54、LOP3 6→30、CS2R 0→9；同时增加了 3 次热 local 读和 3 次写。
整个 kernel 的 ptxas stack 为 32 B；不能因资源表 `LOCAL:0` 就说不存在 spill。

两个预设门槛失败：静态工作量上限 1.01 倍、热 local 至多 1 条。其余原生 payload、
供数/barrier 数、寄存器分配和 IMAD 减少门槛通过，但不足以支持继续投入。
新 range guard 的额外全 K 读取和同步还未计入这个热循环比较，实际运行并不会免费。
**停止此候选，不放宽门槛、不扫描相邻布局、不移植 O3。**

没有候选 GPU 执行、Event、MSE、CV、sanitizer、NCU 或实际驻留测量。
不能写成“延迟增加 20.37%”，也不能写成“实测 MSE 不变”；本轮是编译审计淘汰。
原正式扩展 SHA 保持 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，
旧最佳、转换路径和 5090 后端不变。

原始编译日志、PTX/SASS、坐标与 liveness 见 [冻结证据](evidence/a100_o378_roof_v129/README.md)。
代码：[生成器](../scripts/probe_o78_tensor_factor_codegen.py)、[独立入口](../csrc/sm80/roof_o78_tensor_factor_probe.cu)。

## 瓶颈与可移植性结论

当前优化不能只看“少了几次标量乘法”：小整数外积转给 Tensor Core 的收益，必须覆盖
fragment 构造、额外 MMA、依赖和寄存器成本。当前 A100 实现没有满足这一条件。
这并非证明所有 Tensor Core 系数算法都不可能更快，而是排除这个具体候选，避免继续扫描。

可以迁移的是按执行单元分摊工作并核对总成本的思想；整数 MMA、输入 fragment、原生位宽、
吞吐与资源预算均须按平台重做。此候选未被采用，不能作为“已验证有效的跨平台优化”。
当前已确认的通用策略及 5090/其他平台限制见 [瓶颈与迁移说明](o3_o7_o8_bottleneck_portability_20261007.md)。

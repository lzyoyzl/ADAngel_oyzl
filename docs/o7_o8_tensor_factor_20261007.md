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

## 状态

独立实现/编译门槛准备中；没有候选GPU执行、性能、MSE或默认切换。
代码在[生成器](../scripts/probe_o78_tensor_factor_codegen.py)、[独立入口](../csrc/sm80/roof_o78_tensor_factor_probe.cu)。

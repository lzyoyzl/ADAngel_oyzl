# O0/O1/O2/O3/O4 最终实现与实验结果报告

## 1. 报告范围

实验条件如下：

- GPU：NVIDIA GeForce RTX 5090，SM120a；
- CUDA Toolkit：12.8；
- CUTLASS：4.5.2，commit `db1c288993354c88e551c40c19a8fb93a774a241`；
- 最终 accumulator 和输出：FP32；
- 单 CUDA stream，所有显存与 descriptor 在计时前分配；
- 数据：Llama-2-7B FP16 prefill trace，batch size 为 1，有效 token 数为 4096；
- 层：`0, 6, 12, 18, 24, 31`；
- projection：`q_proj/k_proj/v_proj/o_proj`；
- 样本总数：6 层 × 4 projection = 24；
- 正式运行：warmup 50 次、测量 200 次；
- 转换阶段：`conversion_inner_repeats=100` 批量摊销；
- 稳定性门槛：每条记录的所有计时阶段 `CV<3%`。



## 2. 公共输入与可比性边界

公共 FP16 trace 会在实验前准备成以下格式，公共准备时间不计入任何 variant：

```text
A_int8          [4096,4096]   int8
A_scale         [4096]        fp32

W_mxfp4         [4096,2048]   packed E2M1，K32 scale
W_scale         [4096,128]    UE8M0

W_mxfp4_g128    [4096,2048]   packed E2M1，G128 scale
W_scale_g128    [4096,32]     UE8M0

Y               [4096,4096]   fp32
```

O0/O1/O2 使用同一份 K32 MXFP4 权重；O3/O4 使用同一份 G128 MXFP4/Q4 权重。因此：

- 五者的矩阵尺寸、trace、输出类型、计时方法完全一致，可以比较执行性能；
- O0/O1/O2 之间同时保持 K32 权重量化语义；
- O3/O4 之间同时保持 G128/Q4 数值语义；
- O3/O4 相对 O0 的 MSE 同时包含 K32→G128 分组和 E2M1→Q4 映射造成的误差；
- 本实验的 MSE 是 kernel 输出误差，不等价于模型 perplexity 或下游任务精度。

## 3. 五个最终后端

### 3.1 O0：反量化到 FP16 后执行标准 GEMM

```text
W_mxfp4 + W_scale  -> W_fp16
A_int8 + A_scale   -> A_fp16
A_fp16 @ W_fp16.T  -> Y_fp32
```

实现要点：

- CUDA kernel 完成 MXFP4→FP16 权重反量化；
- CUDA kernel 完成 INT8→FP16 激活反量化；
- cuBLASLt 执行 FP16×FP16 Tensor Core GEMM；
- 使用 `CUBLAS_COMPUTE_32F`，FP32 accumulator 和 FP32 输出；
- algorithm 搜索在计时区间外完成；验收时选择 algorithm ID 21、workspace 0；
- O0 没有自定义固定 CTA tile，tile 和调度由 cuBLASLt 内部算法决定。

O0 是唯一 MSE 主参考，但它本身也是从公共量化数据反量化得到的输出，并不等于原始
FP16 模型输出。

### 3.2 O1：INT8 Tensor Core 与逐 K32 软件 scale

```text
W_mxfp4 -> 2*E2M1 的 INT8 基值
A_int8  -> 保持不变

每个 K32：
    S8×S8 -> S32 IMMA
    INT32 partial -> FP32
    乘 A_scale[row] * W_scale[column,group] / 2
    累加到 FP32 accumulator

最终只写一次 Y_fp32
```

当前 production：

```text
implementation       register_128x64_k64_scale_shared_row_dedup
CTA tile             128×64×64
producer/consumer    1 warp / 16 warps
pipeline             3 stages，2个独立K32/stage
MMA                  m16n8k32 S8×S8->S32
data movement        TMA
schedule             cooperative warp specialization
partial              register
output store         每元素一次
```

关键优化与数值约束：

- 两个相邻 K32 一起进入 K64 pipeline，但仍分别执行 MMA、scale 和 FP32 累加；
- A row scale 在 K 循环外按线程拥有的唯一行加载；
- producer 对每个 CTA/column/group 只解码一次 W scale，并通过 stage-local shared
  memory 分发给 consumer；
- INT32 partial 不再写入 shared/global memory；
- 共享内存仍用于 TMA A/W pipeline 和 W scale 分发；
- 保持 group 0→127 的 FP32 累加顺序。

### 3.3 O2：原生 SM120 MXFP4 block-scaled GEMM

```text
A_int8 * A_scale
    -> MXFP4 E2M1 + UE8M0/K32
    -> CUTLASS SFA layout

W_mxfp4 保持不变
W_scale -> CUTLASS SFB layout

MXFP4 block-scaled MMA -> Y_fp32
```

当前 production：

```text
CTA tile             128×128×256
MMA                  m16n8k64 MXFP4 block-scaled MMA
scale vector         K32，UE8M0
data movement        TMA
schedule             KernelTmaWarpSpecializedCooperative
accumulator/output   FP32/FP32
```

O2 的主要区别是 K32 scale 由原生 block-scaled MMA 消费，不需要像 O1 那样在每个
K32 partial 后执行 INT32→FP32 和软件 scale。O2 的在线代价是把激活重新量化为
MXFP4，并生成/重排 SFA。

### 3.4 O3：Split INT4 Tensor Core

激活按原始 two's-complement 位模式拆分：

\[
A_8=A_{low,u4}+16A_{high,s4}
\]

G128 权重由 MXFP4 通过 RNE 映射到 signed Q4。每个 G128 分别计算：

```text
P_low  = U4(A_low)  × S4(W)
P_high = S4(A_high) × S4(W)
P      = P_low + 16*P_high
Y     += P * A_scale[row] * W_scale_g128[column,group]
```

当前 production：

```text
implementation       n16_k128_cute_ldsm
CTA tile             128×16×128
producer/consumer    1 warp / 16 warps
pipeline             2 stages，1个G128/stage
MMA                  m16n8k64 U4×S4 和 S4×S4
data movement        TMA
schedule             cooperative warp specialization
partial              register
shared→register      explicit LDSM fragment load
accumulator/output   FP32/FP32
dynamic shared       35,072 bytes
```

一个 G128 对每条 low/high 路径各需要两个 K64 INT4 MMA，共四个 INT4 MMA，然后在
寄存器中完成整数重构、G128 scale 和 FP32 累加。

### 3.5 O4：8×4 Bitplane Binary Tensor Core

INT8 激活被拆成 8 个 two's-complement bitplane：

```text
[1, 2, 4, 8, 16, 32, 64, -128]
```

signed Q4 权重被拆成 4 个 bitplane：

```text
[1, 2, 4, -8]
```

每个 G128 对全部 8×4=32 个 plane pair 执行 Binary MMA，并按位权重构：

```text
BMMA = m16n8k128.b1.b1.s32.and.popc
Y   += reconstructed_int_dot * A_scale * W_scale_g128
```

当前 production：

```text
implementation       m64_n64_k512_optimized
CTA tile             64×64×512
producer/consumer    1 warp / 16 warps
warp layout          4×4
pipeline             2 stages，4个独立G128/stage
MMA                  m16n8k128 B1 AND-POPC
BMMA chains          2
B fragment cache     enabled
data movement        TMA
partial              register
accumulator/output   FP32/FP32
dynamic shared       100,480 bytes
```

每个 G128 的四个权重 bitplane fragment 只加载一次，两个独立 accumulator chain 用于
缩短 BMMA 后处理依赖。O4 的 bitplane 生成仍作为独立 GPU conversion 计时；当前实现
对齐 8×4 Bitwise 算术，但不宣称使用 selective fusion。

## 4. 计时和指标定义

### 4.1 Conversion-only

只测配置专用格式转换：

| Variant | 权重转换 | 激活转换 |
|---|---|---|
| O0 | MXFP4→FP16 | INT8→FP16 |
| O1 | E2M1→INT8 基值 | 无，保持 INT8 |
| O2 | natural W scale→SFB | INT8→MXFP4，并生成/repack SFA |
| O3 | G128 MXFP4→Q4 | INT8→U4 low/S4 high |
| O4 | G128 MXFP4→Q4→4 planes | INT8→8 planes |

转换操作只有数微秒到数十微秒。每个外层样本在同一个 CUDA Event 区间内连续执行
100 次，再除以 100，降低 Event 分辨率和偶发调度对短 kernel 的影响。

### 4.2 Compute-only

所有输入转换和静态布局准备均提前完成，只测最终 GEMM。O1 的逐 K32 软件 scale、
O3 的 Split 重构和 O4 的 bitplane 重构属于各自 GEMM 语义，仍计入 compute-only。

### 4.3 Cold end-to-end

从公共 prepared input 出发，直接测量一次完整调用：

```text
cold = 权重转换 + 激活转换 + GEMM
```

### 4.4 Steady-state end-to-end

缓存静态权重转换，只保留在线激活路径和 GEMM：

```text
O0 steady = A反量化 + GEMM
O1 steady = GEMM
O2 steady = A重量化/SFA + GEMM
O3 steady = A Split + GEMM
O4 steady = A bitplanes + GEMM
```

转换组件与端到端 total 是独立测量的。组件来自 100 次批量摊销，total 来自一次真实
路径，且每列分别取 median。因此不能要求：

```
median(W)+median(A)+median(GEMM)=median(total)
```


### 4.6 MSE

O0 是唯一参考：

\[
MSE(O_i,O0)=\frac{1}{MN}\sum_{m,n}(Y_i[m,n]-Y_{O0}[m,n])^2
\]

两侧输出转 FP64 后做 reduction，避免指标 reduction 自身引入明显误差。每个样本只用
compute-only 输出计算一次 MSE，再把该值关联到同一 sample/variant 的四种 mode；四个
mode 不是四次独立 MSE 观测。



## 6. 24 个真实样本的最终性能

以下四张计时表使用相同的两层汇总口径：每个样本先对 200 次测量计算mean/median/P5/P95/IQR/CV，再对 24 个样本的同一统计列取中位数。

本节保留的是全部480条记录均通过 `CV<3%` 的稳定 run；其中 O3 数字对应显式 LDSM
晋升前的 `n16_k128`。新 O3 production `n16_k128_cute_ldsm` 已重新运行24样本并
确认 metadata/MSE，但共享 GPU 外部进程使451/480条记录 CV 超标，因此不以受干扰
绝对延迟覆盖本节。新 O3 的性能收益以两轮同进程配对 A/B 的中位数及置信区间报告。

表中的“最大 CV”是24 个样本中最大的单样本 CV，而不是中位数。“相对 O0”定义为
`O0 median latency / variant median latency`：大于 1 表示快于 O0，小于 1 表示慢于
O0。


### 6.1 Conversion-only：转换开销

该模式只测配置专用转换，不测正式 GEMM。`W`/`A` 分别表示权重和激活转换，`Total`
是完整转换序列在另一个批量 Event 区间内的直接测量，不由 W/A 的统计量相加得到。

| Variant | Stage | Mean ms | Median ms | P5 ms | P95 ms | Max CV % |
|---|---|---:|---:|---:|---:|---:|
| O0 | W | 0.017449 | 0.017455 | 0.017419 | 0.017466 | 0.780593 |
| O0 | A | 0.016217 | 0.016207 | 0.016196 | 0.016248 | 0.782036 |
| O0 | Total | 0.035605 | 0.035777 | 0.034930 | 0.035898 | 1.216437 |
| O1 | W | 0.032821 | 0.032821 | 0.032791 | 0.032852 | 0.436694 |
| O1 | Total | 0.032821 | 0.032821 | 0.032791 | 0.032852 | 0.436694 |
| O2 | W | 0.002083 | 0.002086 | 0.002066 | 0.002091 | 0.830007 |
| O2 | A | 0.069118 | 0.069422 | 0.067031 | 0.069526 | 1.525502 |
| O2 | Total | 0.071251 | 0.071246 | 0.071182 | 0.071301 | 0.300297 |
| O3 | W | 0.029899 | 0.029893 | 0.029870 | 0.029925 | 0.489485 |
| O3 | A | 0.016328 | 0.016325 | 0.016315 | 0.016345 | 0.766033 |
| O3 | Total | 0.046228 | 0.046227 | 0.046196 | 0.046268 | 0.316855 |
| O4 | W | 0.042014 | 0.042017 | 0.041996 | 0.042038 | 0.347186 |
| O4 | A | 0.072893 | 0.072890 | 0.072870 | 0.072912 | 0.375371 |
| O4 | Total | 0.114911 | 0.114902 | 0.114872 | 0.114935 | 0.237376 |

### 6.2 Compute-only（GEMM kernel-only）

输入已经转换完成，只测完成矩阵乘所需的正式计算 kernel。O1 的逐 K32 scale、O3 的
Split 重构和 O4 的 Bitplane 重构仍属于该 kernel，因此包含在本表中。

| Variant | Mean ms | Median ms | P5 ms | P95 ms | Max CV % | 相对 O0 | Median MSE vs O0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| O0 | 0.767158 | 0.767520 | 0.765418 | 0.768784 | 1.682949 | 1.000× | 0 |
| O1 | 0.622808 | 0.622136 | 0.608831 | 0.641196 | 2.043037 | 1.234× | `9.8234e-09` |
| O2 | 0.137972 | 0.138512 | 0.136496 | 0.139008 | 1.312743 | 5.541× | `3.7960e-03` |
| O3 | 2.223679 | 2.223352 | 2.217901 | 2.228738 | 0.550200 | 0.345× | `6.6530e-03` |
| O4 | 7.640403 | 7.632488 | 7.613085 | 7.686460 | 0.395309 | 0.101× | `6.6530e-03` |



### 6.3 Cold：未缓存配置专用权重的直接端到端路径

Cold 在同一个 Event 区间直接执行一次权重转换、一次激活转换和一次正式 GEMM。下表
只统计该直接 total，不使用 Conversion-only 与 Compute-only 的独立统计量相加。

| Variant | Mean total ms | Median total ms | P5 ms | P95 ms | Max CV % | 相对 O0 |
|---|---:|---:|---:|---:|---:|---:|
| O0 | 0.839026 | 0.838944 | 0.836799 | 0.842530 | 0.588579 | 1.000× |
| O1 | 0.668963 | 0.668376 | 0.652546 | 0.685446 | 1.671792 | 1.255× |
| O2 | 0.205516 | 0.204984 | 0.204256 | 0.208419 | 1.383190 | 4.093× |
| O3 | 2.300093 | 2.300128 | 2.297968 | 2.302978 | 0.721558 | 0.365× |
| O4 | 7.758471 | 7.758064 | 7.744117 | 7.767762 | 0.269155 | 0.108× |



### 6.4 Steady-state：缓存静态权重后的直接端到端路径

Steady-state 缓存配置专用权重，只直接测量在线激活转换和正式 GEMM；O1 的公共激活
已经是 INT8，因此其 Steady-state 基本就是正式 O1 kernel。

| Variant | Mean total ms | Median total ms | P5 ms | P95 ms | Max CV % | 相对 O0 |
|---|---:|---:|---:|---:|---:|---:|
| O0 | 0.806827 | 0.806664 | 0.804473 | 0.809280 | 0.317346 | 1.000× |
| O1 | 0.622950 | 0.622192 | 0.607854 | 0.640167 | 1.899900 | 1.296× |
| O2 | 0.214247 | 0.214224 | 0.212112 | 0.216352 | 1.234807 | 3.766× |
| O3 | 2.264220 | 2.263328 | 2.260930 | 2.267360 | 0.773832 | 0.356× |
| O4 | 7.712770 | 7.712608 | 7.694377 | 7.724244 | 0.304157 | 0.105× |

Cold/Steady 的 total 与 Conversion-only/GEMM-only 来自不同 Event 区间、缓存状态和统计分布，不能把后两者的 median 相加来重建端到端 total。

### 6.5 相对 O0 的输出 MSE

| Variant | Median | Mean |
|---|---:|---:|
| O0 | 0 | 0 |
| O1 | `9.8234e-09` | `1.1141e-08` |
| O2 | `3.7960e-03` | `5.7348e-03` |
| O3 | `6.6530e-03` | `7.5788e-03` |
| O4 | `6.6530e-03` | `7.5788e-03` |

O3 和 O4 的 24 个逐样本 MSE 完全相同。这说明两者精确实现了同一个 G128/Q4 整数
点积，只是分别通过 Split INT4 和 8×4 Bitplane Binary Tensor Core 得到结果。

## 7. 性能差异分析

### 7.1 为什么 O1 能快于 O0

O0 在转换后执行一次高度优化的 cuBLASLt FP16 GEMM，已经是很强的基线。O1 仍能在
GEMM-only 上达到 `1.234×` 加速，主要因为：

- INT8 Tensor Core 每条指令处理的数据密度高于 FP16；
- TMA 和 cooperative warp specialization 隐藏 A/W tile 搬运；
- INT32 partial 始终保留在寄存器；
- A scale 被移出 128 个 group 循环；
- W scale 每 CTA/column/group 只解码一次；
- 每个输出元素最终只写全局显存一次。

O1 没有获得接近 O2 的速度，因为普通 INT8 MMA 不接受 UE8M0 scale。每个 K32 仍必须
执行 INT32→FP32、读取/解码 scale、乘 `A_scale*W_scale/2` 并按顺序累加，这些都是
O1 GEMM 内无法省略的软件工作。

### 7.2 为什么 O2 最快

O2 的 `0.138512 ms` 是五种配置中最低的 GEMM-only 延迟。核心原因不是单纯“4 bit
比 8 bit 小”，而是 SM120 MXFP4 block-scaled MMA 在硬件指令中直接消费 K32 UE8M0
scale：

- 不生成 INT32 partial；
- 不在每个 K32 后执行软件 rescale；
- K=256 的主循环可使用成熟的 CUTLASS TMA cooperative pipeline；
- `128×128` 输出 tile 提供较高的数据复用与并行覆盖。

O2 的代价集中在在线激活转换：A conversion 为 `0.069422 ms`，明显高于 O0/O3 的
简单激活转换。不过即使计入该代价，O2 的 cold/steady total 仍分别比 O0 快
`4.093×/3.766×`。

O2 的 steady median 略高于 cold median，并不表示 steady 路径包含更多工作。两种 total
独立直接测量、分别取 median，且都处于约 0.2 ms 的动态 Boost 区间；因此 mode 间可能
出现小幅顺序变化，不能用阶段 median 相加反推。

### 7.3 为什么 O3 慢于 O0/O1

O3 虽然使用 INT4 Tensor Core，但每个 G128 不能通过一次 MMA 完成：

```text
low U4 路径：  2个 K64 MMA
high S4 路径： 2个 K64 MMA
软件重构：     P_low + 16*P_high
软件缩放：     A_scale * W_scale_g128
```

也就是说，每个 G128 至少需要四个 INT4 MMA，加上两路 partial 重构和 FP32 scale。
相比之下，O1 在同一个 G128 范围内执行四个 K32 INT8 MMA，但 O1 的当前 CTA、scale
共享和寄存器路径更加适合这一矩阵形状；O3 还需要维护 low/high 两套数据流。O3 的
`128×16` 输出 tile 在 N 方向覆盖较窄，也限制了 B tile 和 scale 的复用范围。

因此，低位宽 Tensor Core 并不自动意味着端到端更快；分解所增加的指令数和软件重构
超过了 INT4 数据密度带来的收益。

### 7.4 为什么 O4 最慢

O4 的单次 BMMA 输入只有 1 bit，但一个数值点积由多个 bitplane 点积重构。每个 G128：

```text
8个激活plane × 4个权重plane = 32个BMMA
```

当前使用 `m16n8k128`，每条 BMMA 输出 `16×8` 个 popcount 结果。随后还需要：

- 根据激活和权重 two's-complement 系数做带符号重构；
- 管理多个 INT32 accumulator chain；
- 对每个 G128 应用 FP32 scale；
- 在转换阶段生成并存储 12 组 bitplane 数据。

作为数量级对比，在同一个 G128 上：

```text
O1：4个 K32 INT8 MMA
O3：4个 K64 INT4 MMA + 两路重构
O4：32个 K128 Binary MMA + 32项带符号重构
```

因此 O4 的主要瓶颈是精确 8×4 Bitwise 算术要求的指令倍数，而不是显存写回或 partial
落盘。TMA、K512 pipeline、B fragment cache、双依赖链和寄存器 partial 已经减少了
搬运与调度成本，但不能在不改变数学语义的前提下消除 32 个 plane pair。

### 7.5 精度差异来自哪里

- **O1**：E2M1→INT8 基值是精确映射，仍使用原 K32 scale；与 O0 的差异主要来自
  FP16 GEMM 与分组 INT32/FP32 累加的舍入路径，因此 MSE 约为 `1e-8`。
- **O2**：权重保持原 K32 MXFP4，但激活被再次量化到 MXFP4，产生约 `3.8e-3`
  median MSE。
- **O3/O4**：使用更粗的 G128 scale，并将权重映射为 signed Q4，median MSE 上升到
  `6.65e-3`。两者 MSE 完全一致，说明 Binary 分解没有引入额外数值近似。

## 8. 最终实验结论

1. **性能最高的是 O2。** 原生 MXFP4 block-scaled MMA 能在硬件内部处理 K32 scale，
   GEMM-only 比 O0 快 `5.541×`，cold/steady 仍快 `4.093×/3.766×`。
2. **O1 提供了最好的性能—误差折中。** 它的 GEMM-only 比 O0 快 `1.234×`，而 MSE
   median 仅为 `9.82e-9`；代价是 kernel 必须保留逐 K32 软件 scale。
3. **O3 正确实现了 Split INT4，但没有获得低位宽理论上的直接加速。** low/high 两条
   路径、四个 INT4 MMA 和软件重构使其 GEMM-only 约为 O0 的 `2.90×` 延迟。
4. **O4 正确使用了 B1 Tensor Core，但精确 8×4 bitplane 展开具有结构性开销。**
   32 个 BMMA/组使其 GEMM-only 约为 O0 的 `9.94×` 延迟，不能用“1 bit”直接推断
   它应快于 INT4/INT8。
5. **O3/O4 是同一数值语义的两条实现路径。** 二者逐样本 MSE 完全相同，性能差异
   反映 Split 与 Bitwise Tensor Core 分解成本，而不是量化误差差异。
6. **最终汇报必须同时给出 GEMM、转换、端到端和 MSE。** 仅看 Tensor Core 位宽或
   GEMM-only 会遗漏 O2/O4 较高的在线激活转换成本；仅看 MSE 又无法解释执行效率。

## 9. 复现命令

```bash
python -m adangel doctor --require-native

python scripts/validate_o0.py
python scripts/validate_o1.py
python scripts/validate_o2.py
python scripts/validate_o3.py
python scripts/validate_o4.py

python -m pytest \
  tests/integration/test_sm120_o0.py \
  tests/integration/test_sm120_o1.py \
  tests/integration/test_sm120_o2.py \
  tests/integration/test_sm120_o3_o4.py \
  -q --run-sm120

python -m adangel run \
  --config configs/experiment/o0_o1_o2_o3_o4_4096.yaml \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/rtx5090_o0_o4_k512 \
  --require-native

python scripts/analyze_results.py \
  --run runs/rtx5090_o0_o4_k512 \
  --output reports/rtx5090_o0_o4_k512 \
  --stability-policy strict

EXTENSION_DIR=$(python -c \
  "import torch,pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit_o0_o4_final
```

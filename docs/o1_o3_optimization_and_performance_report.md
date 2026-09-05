# O1/O3 当前后端优化与性能分析

## 1. 文档目的

本文面向实验汇报，说明 RTX 5090（SM120a）上 O1 与 O3 **当前 production 后端**的：

- 数值语义与矩阵计算路径；
- 已经落实在正式内核中的优化手段；
- 24 个 Llama-2-7B 真实 prefill 样本的性能与 MSE；
- O3 未达到“O1 一半吞吐”目标的原因。

本文只描述当前最终方案，不列出候选内核、消融过程或历史迭代结果。

## 2. 公共实验条件

正式实验固定为：

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，SM120a |
| CUDA | 12.8 |
| CUTLASS | 4.5.2，commit `db1c288993354c88e551c40c19a8fb93a774a241` |
| 矩阵计算 | `Y = A @ W.T` |
| 矩阵尺寸 | `M=N=K=4096` |
| 输出 | FP32 accumulator，FP32 output |
| 数据 | Llama-2-7B FP16 prefill trace |
| 样本 | 6 层 × 4 个 projection，共 24 个样本 |
| 正式计时 | warmup 50，repeats 200，单 CUDA stream |
| 转换计时 | 一个 CUDA Event 区间内执行 100 次后除以 100 |
| 稳定性 | 每个计时 stage 要求 `CV<3%` |

公共准备得到逐行对称量化的 INT8 激活：

```text
A_int8   [4096,4096] int8
A_scale  [4096]      fp32
```

O1 使用 K32 MXFP4 权重，O3 使用 G128 MXFP4/Q4 权重。两者采用相同的 trace、矩阵尺寸、
输出类型、计时方法和 O0 参考输出，但权重分组与整数计算语义不同。

## 3. O1：原生 INT8 IMMA 与逐 K32 软件 scale

### 3.1 数值路径

O1 从公共 K32 MXFP4 权重出发，将 E2M1 private value 精确映射为 `2×E2M1` 的 INT8
基值：

```text
E2M1 magnitude  0, 0.5, 1, 1.5, 2, 3, 4, 6
INT8 base       0, 1,   2, 3,   4, 6, 8, 12
```

该转换不会产生新的权重量化误差。由于基值扩大了 2 倍，运行时 scale 中需要除以 2。
第 `g` 个 K32 分组的计算为：

\[
P^{(g)}_{m,n}=\sum_{k=32g}^{32g+31}
A_{\mathrm{int8},m,k}W_{\mathrm{int8},n,k},
\]

\[
Y^{O1}_{m,n}=\sum_{g=0}^{127}
FP32(P^{(g)}_{m,n})
\left(A\_scale_m\frac{decode(W\_scale_{n,g})}{2}\right).
\]

普通 INT8 MMA 不接受 UE8M0 block scale，因此 128 个 K32 分组必须分别形成 INT32
partial、分别应用 scale，再按分组顺序累加到 FP32。

### 3.2 当前 production 配置

```text
implementation       register_128x64_k64_scale_shared_row_dedup
kernel symbol        adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup
CTA tile             128 x 64 x 64
threads/CTA          544
producer/consumer    1 warp / 16 warps
pipeline             3-stage TMA
groups/stage         2 个数值独立的 K32 group
MMA                  CuTe m16n8k32, S8 x S8 -> S32
partial storage      register
accumulator/output   FP32 / FP32
```

一个 CTA 最终负责 `Y[128,64]`。每个 pipeline stage 搬运 `A[128,64]` 和
`W[64,64]`，即两个相邻 K32 group。两个 group 共享一次 K64 数据搬运和 pipeline
同步，但仍各自执行 INT8 MMA、应用自己的 W scale，并独立累加。

### 3.3 当前优化手段

#### TMA 与 cooperative warp specialization

- producer warp 使用 TMA 将后续 A/W tile 搬入三阶段 shared-memory pipeline；
- 16 个 consumer warp 负责 CuTe MMA 与后处理；
- 数据搬运和计算重叠，shared memory 只承担 A/W staging 与 W scale 分发。

#### INT32 partial 全程保留在寄存器

CuTe `partition_C(identity_tensor)` 为每个 accumulator register 提供明确的逻辑
`(row,column)` 坐标。INT32 MMA partial 不写入 shared/global memory，而是在寄存器中：

```text
INT32 partial
  -> FP32
  -> 乘 A row scale 和 W column/group scale
  -> FMA 到 FP32 accumulator
```

这消除了 partial 的 shared-memory 写回、读回和围绕重分发的同步；完成全部 K32 group
后，每个输出元素只写全局显存一次。

#### A row scale 循环外去重

每个 consumer 线程拥有的 accumulator fragment 只涉及两个唯一输出行。对应的两个
`A_scale` 在 K 循环前各加载一次并保存在寄存器中，而不是在 128 个 K32 group 中重复
加载。

#### W scale CTA 内共享

producer 对每个 `(CTA column, K32 group)` 只解码一次 UE8M0 W scale，将
`decode(W_scale)/2` 写入与 A/W 数据相同生命周期的 stage-local shared memory。
consumer 从 shared memory 取得所需列 scale，并通过 warp shuffle 匹配 accumulator
列坐标，避免每个 consumer warp 重复访问和解码全局 W scale。

#### K64 pipeline 保留 K32 数值语义

每个 stage 一次准备两个相邻 K32 group，减少 pipeline 状态推进和同步次数；但两个
partial 不会在应用 scale 前合并，因此没有改变 K32 独立缩放语义，也没有改变 O1 的
MSE 定义。

#### 审计约束

正式 kernel 的同一 SASS 函数中包含 TMA 与 signed INT8 IMMA；资源审计满足
`STACK=0, LOCAL=0`，不存在 `LDL/STL` spill。`global_partial_buffer=false`，每个输出
元素只进行一次最终 store。

## 4. O3：Split Q4/G128 与 sub-byte MMA 语义

### 4.1 权重与激活表示

O3 权重先按 G128 量化为 MXFP4 E2M1+UE8M0，再将组内 E2M1 private value 使用
round-to-nearest、ties-to-even 映射为 signed Q4：

```text
E2M1 magnitude  0, 0.5, 1, 1.5, 2, 3, 4, 6
Q4 magnitude    0, 0,   1, 2,   2, 3, 4, 6
```

该映射是 O3 相对 O0 的主要量化误差来源之一。激活保持原始 INT8 two's-complement
数值，但为匹配 4-bit MMA 接口拆成：

```cpp
uint8_t u = static_cast<uint8_t>(a);
low_u4  = u & 0x0f;
high_s4 = sign_extend_4bit(u >> 4);
```

恒等式严格成立：

\[
A_{int8}=A_{low,u4}+16A_{high,s4}.
\]

激活 Split 本身不引入近似误差。

### 4.2 每个 G128 的计算

对每个 G128 group：

\[
P_{low}=MMA_{U4\times S4}(A_{low},W_{q4}),
\]

\[
P_{high}=MMA_{S4\times S4}(A_{high},W_{q4}),
\]

\[
P_g=P_{low}+16P_{high},
\]

\[
Y^{O3}_{m,n}=\sum_{g=0}^{31}FP32(P_g)
\left(A\_scale_m\,decode(W\_scale^{G128}_{n,g})\right).
\]

一个 G128 由两个 K64 subgroup 覆盖，因此每个 group 包含两条 low 和两条 high
逻辑 MMA 路径。各 G128 的 W scale 仍分别应用，不能跨组提前合并整数 partial。

### 4.3 当前 production 配置

```text
implementation       m64_n32_k128_aligned_factor_16w
kernel symbol        adangel_o3_split_tma_ws<O3M64N32K128AlignedFactor16WConfig>
CTA tile             64 x 32 x 128
threads/CTA          544
producer/consumer    1 warp / 16 warps
pipeline             2-stage TMA
groups/stage         1 个 G128 group
PTX MMA semantics    m16n8k64 U4xS4 + S4xS4
partial storage      register
accumulator/output   FP32 / FP32
G128 loop            fully unrolled, factor 32
```

16 个 consumer warp 采用 `4×4` warp layout，每个 warp 负责一个 `16×8` 输出 fragment，
共同覆盖 `Y[64,32]`。

### 4.4 当前优化手段

#### TMA 与 cooperative warp specialization

producer warp 使用 TMA 将 packed `A_low`、`A_high` 和 Q4 W 的下一个 G128 tile 搬入
两阶段 shared-memory pipeline；consumer warp 执行两路 MMA、整数重构、缩放和 FP32
累加，使数据搬运与计算重叠。

#### 显式 LDSM fragment load

consumer 通过 CUTLASS/CuTe 的 `SM75_U32x4_LDSM_N` 和 `SM75_U32x2_LDSM_N` wrapper，
按照已验证的 `m16n8k64` lane/fragment 映射将 packed 4-bit operand 从 shared memory
装入寄存器。该路径避免把 MMA operand 逐元素标量化后再手工猜测 lane 布局。

#### INT32 partial 与 FP32 accumulator 留在寄存器

low/high 两路 INT32 partial 在寄存器中重构为 `P_low + 16×P_high`，不使用全局或 shared
partial buffer。所有 G128 完成后只写一次 FP32 输出。

#### row A scale 移出 G128 主循环

`A_scale[row]` 与 G128 group 无关。正式 kernel 先在寄存器中完成 32 个 G128 的
`partial × W_scale` 累加，再统一乘一次 row A scale，减少主循环内重复 FP32 乘法。
运算顺序经过 MSE 回归验证，没有引入新的量化误差。

#### W scale CTA 内一次解码

producer 对每个 `(CTA column, G128 group)` 解码一次 UE8M0 W scale，并放入
stage-local shared memory供 consumer 使用，避免各 consumer warp 重复访问全局 scale。

#### 正式尺寸对齐快路径

`4096×4096×4096` 能被 `64×32×128` tile 完整整除。production fast path 删除不需要的
M/N tail 条件判断；非对齐的公共 API 输入自动使用带边界检查的安全 kernel。

#### 32 个 G128 pipeline group 完全展开

正式 K=4096 恰好包含 32 个 G128 group。主 producer/consumer 循环采用展开因子 32，
减少循环控制、整数地址计算和分支指令。该优化不改变每组独立缩放和 low/high Split
算术。

#### 审计约束

正式 PTX entry 同时包含 TMA、U4×S4 MMA 与 S4×S4 MMA 语义；资源审计为
`REG=56, STACK=0, LOCAL=0`，无寄存器 spill。需要特别强调：这些是 PTX 接口语义，
并不等同于 SM120 上存在独立的原生 INT4 SASS 指令。

## 5. 当前性能结果

### 5.1 同一时段的直接对比

最终同步后执行的 `runs/final_b10ed0b_o0_o4` 使用同一进程、同一批 24 个样本和交错
variant 顺序。全部 480 条记录满足 `CV<3%`，因此该 run 是当前 O1/O3 最直接的横向
证据。

| 指标 | O1 production | O3 production |
|---|---:|---:|
| Compute-only 跨样本 median | 0.623776 ms | 2.065672 ms |
| 等效吞吐量 | 220.33 TFLOP/s | 66.53 TFLOP/s |
| O3/O1 逐样本等效吞吐比 median | — | 30.526% |

等效吞吐按同一个逻辑 GEMM 的 `2MNK/time` 计算，只用于比较完成相同输出矩阵的速度，
不表示 O3 实际发射的物理指令与标准 GEMM 相同。

### 5.2 转换与端到端延迟

转换 kernel 未随当前 O1/O3 GEMM production 改变。以下是已验收的 24 样本跨样本
median；O1 与 O3 的端到端数字来自各自稳定 production 运行，未将不同时段的绝对值
用于严格配对加速判断。

| Variant | W conversion | A conversion | Conversion total |
|---|---:|---:|---:|
| O1 | 0.032821 ms | 无，A 保持 INT8 | 0.032821 ms |
| O3 | 0.029893 ms | 0.016325 ms | 0.046227 ms |

| Variant | Cold total | Steady-state total | 数据口径 |
|---|---:|---:|---|
| O1 | 0.668376 ms | 0.622192 ms | O1 稳定 24 样本 production run |
| O3 | 2.130128 ms | 2.093312 ms | 最终同步后的 24 样本交错 run |

其中：

- `cold` 包含当前 variant 的 W 转换、A 转换和 GEMM；
- `steady-state` 缓存静态权重转换，只保留在线激活路径与 GEMM；
- O1 的公共激活已经是 INT8，因此 steady-state 基本等于 O1 GEMM；
- O3 steady-state 仍包含在线 `INT8 -> low U4/high S4` Split。

conversion-only 使用批量摊销，而 cold/steady-state total 使用完整路径的单次直接计时，
因此不能把各阶段 median 简单相加来重构 total。

### 5.3 MSE

O0 FP32 输出是唯一主参考。每个样本计算：

\[
MSE(O_i,O0)=\frac{1}{MN}\sum_{m,n}
\left(Y^{O_i}_{m,n}-Y^{O0}_{m,n}\right)^2,
\]

并将两侧输出转为 FP64 后执行 reduction。24 样本结果为：

| Variant | MSE median vs O0 | MSE mean vs O0 | 主要误差来源 |
|---|---:|---:|---|
| O1 | `9.8234e-09` | `1.1141e-08` | FP16 GEMM 与逐 K32 INT32/FP32 累加路径的舍入差异 |
| O3 | `6.653010195e-03` | `7.578844400e-03` | K32→G128 分组变化与 E2M1→Q4 RNE 映射 |

O1 的 E2M1→INT8 base 映射是精确的，因此 MSE 接近 0；O3 的 activation Split 是严格
恒等分解，本身也不增加误差。O3 较大的 MSE 来自更粗的 G128 权重 scale 和 Q4 映射，
而不是 TMA、tile、LDSM 或 SASS lowering。

## 6. 当前性能瓶颈分析

### 6.1 O1 为什么能够达到约 0.62 ms

O1 的 S8×S8 MMA 最终编译为 SM120 可直接执行的 INT8 IMMA。每个 K32 group 的核心
整数点积只需要一条逻辑 `m16n8k32` 路径；TMA pipeline、较大的 `128×64` 输出 tile、
CTA 级 W scale 共享、A scale 去重和寄存器 partial 又摊薄了外围开销。

O1 仍不是纯粹的标准 INT8 GEMM。普通 INT8 MMA 不接受 UE8M0 scale，所以每个 K32
仍必须执行：

```text
INT32 -> FP32
乘 A_scale * W_scale / 2
FMA 到 FP32 accumulator
```

这部分软件 scale 是 O1 的剩余主要成本，但当前实现已经避免了 partial 落盘和重复 scale
加载。

### 6.2 O3 的目标定义

若把“O3 达到 O1 一半吞吐”定义为完成相同 `4096³` 逻辑 GEMM 时吞吐至少为 O1 的
50%，那么延迟必须不超过 O1 的 2 倍：

```text
同一时段 O1 median            = 0.623776 ms
O3 目标延迟上限               = 1.247552 ms
当前 O3 median                = 2.065672 ms
当前 O3/O1 等效吞吐比 median = 30.526%
```

因此当前 O3 距离目标仍需要约 `1.66×` 的额外加速。

### 6.3 O3 无法达到目标的首要原因：PTX INT4 语义没有变成原生 INT4 SASS

这个问题不是根据 O3 延迟较高作出的猜测，而是通过“正式性能异常、NCU 排除内存瓶颈、
正式 kernel 精确入口反汇编、独立 MMA 微基准”四层证据逐步确认的。

#### 6.3.1 从性能计数器发现异常

如果 SM120 能把两路 4-bit PTX 直接映射成具有理想 INT4 吞吐的物理 MMA，那么 O3
即使有 Split 和 scale 后处理，也不应出现数十亿条通用整数指令。实际对当前 production
采集 NCU 得到：

| 证据 | 当前 O3 production | 判断 |
|---|---:|---|
| NCU duration | 1.933472 ms | 与 CUDA Event 约 2.0 ms 的结果一致 |
| 动态 SASS | 2,507,181,638 | 指令规模远高于单纯的 MMA 主循环 |
| SM throughput | 93.81% | SM 执行端已经接近繁忙 |
| DRAM throughput | 2.00% | 排除 DRAM 带宽瓶颈 |
| Tensor pipe active | 14.20% | Tensor Core 并非大部分时间的主导执行管线 |
| Math-pipe throttle | 4.17 cycles/issued instruction | 数学/整数执行资源存在明显竞争 |
| Resource usage | `REG=56, STACK=0, LOCAL=0` | 排除寄存器 spill/local memory 为主因 |

这一组数据首先指出：O3 的时间主要消耗在 SM 内部执行工作，而不是等待 TMA、DRAM 或
local-memory spill；同时 Tensor pipe 利用率并没有随高 SM throughput 一起升高，因此
需要继续检查 MMA 前后的整数指令。

#### 6.3.2 对正式 production 的同入口 PTX/SASS 审计

O3 源码使用 CuTe 的 sub-byte MMA wrapper。编译为 `sm_120a` 后，审计脚本先根据正式
metadata 锁定：

```text
adangel_o3_split_tma_ws<O3M64N32K128AlignedFactor16WConfig>
```

随后只比较这个 production entry 的 PTX 与 SASS，避免误命中 probe、测试 kernel 或
其他候选实现。PTX 中确实保留实验要求的两路语义：

```text
U4 x S4 -> S32
S4 x S4 -> S32
```

同一 entry 也包含 TMA load，证明审计的正是正式 fused tiled kernel，而不是独立的
MMA probe。但是 SASS 中没有与上述两条 PTX 一一对应的标准原生 INT4 opcode，实际
出现的是：

```text
IMMA.16832.U8.S8
IMMA.16832.S8.S8
+ 大量 LOP3 / SHF / IMAD
```

这里 PTX 的逻辑 shape 是 `m16n8k64`，而 SASS 显示 `IMMA.16832`，是因为一个逻辑 K64
sub-byte MMA 被 ptxas 分解为多个物理 K32 INT8 IMMA，并辅以 packed nibble 的拆取、
符号扩展、lane 排列和 operand 构造。项目因此在正式 metadata 中明确记录：

```text
ptx_mma_semantics = U4xS4_and_S4xS4
native_int4_sass  = false
sass_lowering     = U8xS8_and_S8xS8_IMMA_plus_bit_operations
```

`native_int4_sass=false` 不是“kernel 没有按 O3 算法计算”或“审计失败”。它表示 PTX
接口仍精确实现 O3 的 U4/S4 数学语义，但 CUDA 12.8 为 SM120 生成的实际机器指令没有
提供预期的原生 INT4 吞吐路径。

正式入口的审计可以用以下命令复现：

```bash
EXTENSION_DIR=$(python -c \
  "import torch,pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit
cat reports/audit/summary.txt
```

审计要求 PTX 的同一 entry 同时命中 TMA、U4×S4 与 S4×S4；SASS 同一函数必须无
`LDL/STL`，并将实际 lowering 写入 summary。

#### 6.3.3 用最小 MMA 微基准排除 TMA、scale 和 tile 的影响

为了确认上述指令膨胀不是正式 GEMM 的 TMA、G128 scale、边界判断或 CTA tile 造成，
项目另外构造了只在寄存器中重复执行 MMA 的最小 kernel。该实验不进行 trace 转换、
TMA 搬运、全尺寸矩阵调度或输出 epilogue，只比较相同环境下的 U4×S4、S4×S4 和
S8×S8 MMA lowering。

四条独立 accumulator chain 的逻辑吞吐结果为：

| PTX shape | U4×S4 logical TOPS | S4×S4 logical TOPS | S8×S8 logical TOPS |
|---|---:|---:|---:|
| `m16n8k64` | 163.67 | 78.58 | 744.62 |
| `m16n8k32` | 138.68 | 72.93 | 372.44 |
| `m8n8k32` | 83.42 | 53.28 | 174.65 |

O3 production 使用的 `m16n8k64` 已经是三个受测 sub-byte shape 中最快的，但其
U4×S4 和 S4×S4 逻辑吞吐分别只有 S8×S8 对照的约 `22.0%` 和 `10.6%`。因此，即使
完全移除 TMA 与 scale，当前 U4/S4 编译路径仍没有表现出“INT4 吞吐约为 INT8 两倍”
的特征。

微基准反汇编进一步统计出一条逻辑 PTX MMA 的典型核心展开：

| PTX 语义 | 典型 lowering 核心 |
|---|---|
| U4×S4 | `2 IMMA + 38 LOP3 + 20 SHF + 14 IMAD` |
| S4×S4 | `2 IMMA + 90 LOP3 + 48 SHF + 42 IMAD` |

S4×S4 high 路径的符号处理尤其昂贵。微基准中的 SASS 仍然是 U8/S8 IMMA 加
`LOP3/SHF/IMAD`，并且同样满足 `STACK=0, LOCAL=0`。这把根因限定在 legacy
sub-byte PTX 的 lowering，而不是正式 O3 的外围数据流。

复现命令为：

```bash
bash scripts/run_o3_mma_lowering_microbenchmark.sh \
  reports/o3_mma_lowering
```

该脚本生成 CUDA Event 性能、PTX、SASS 和 resource usage；这些数据只用于诊断指令
路径，不替代正式 24 样本性能或 MSE。

#### 6.3.4 正式 kernel 动态指令与延迟下界闭环

按照当前 O3 每个 G128、每 consumer warp 需要两条 low K64 和两条 high K64 逻辑路径
计算，上述 MMA lowering 核心至少贡献约 `2,147,483,648` 条动态 SASS，占正式 NCU
实测 `2,507,181,638` 条的 `85.65%`。这说明优化 TMA、scale load、循环控制和 epilogue
只能影响剩余约 `14.35%` 的动态指令，无法消除占主导的 sub-byte lowering。

若极其乐观地假设其余工作全部免费，并按当前执行速率线性缩放，O3 的估计延迟下界仍
约为 `1.704 ms`，高于达到 O1 一半吞吐所需的 `1.247552 ms`。因此“当前严格 O3
语义和工具链下不能达到目标”的结论同时得到：

1. 正式 CUDA Event 性能；
2. 正式 kernel NCU；
3. PTX/SASS 同入口反汇编；
4. 排除外围因素的 MMA 微基准；
5. 动态指令占比和延迟下界。

CUDA 13.1 的独立复核也没有改变该 lowering：U4×S4、S4×S4 微基准相对 CUDA 12.8
分别约为 `1.010×` 和 `0.987×`，属于测量波动，不能作为迁移工具链即可解决问题的
证据。

### 6.4 O3 的算法结构本身也比 O1 复杂

在相同 G128 范围内：

```text
O1：4 个独立 K32 INT8 MMA + 4 次 K32 scale
O3：2 个 low K64 MMA + 2 个 high K64 MMA
    + low/high 整数重构
    + 1 次 G128 scale
```

若硬件真正提供接近两倍 INT8 吞吐的标准 INT4 MMA，O3 才有机会接近“一半 O1吞吐”
目标；当前 SASS lowering 不但没有提供该优势，还为 low/high 两路引入大量通用整数指令。

### 6.5 其他瓶颈排除

第 6.3 节的 NCU、资源审计和微基准还排除了三个容易混淆的解释：

- **不是 DRAM 带宽不足**：DRAM throughput 仅约 2%，而 SM throughput 已达 93.81%；
- **不是寄存器 spill**：正式 kernel 为 `STACK=0, LOCAL=0`，SASS 中没有 `LDL/STL`；
- **不是格式转换过慢**：O3 conversion total 约 `0.046 ms`，远小于约 `2.0 ms` 的
  compute-only GEMM。转换会影响 cold/steady-state，但不能解释 compute-only 未达目标。

因此当前优化判断应聚焦于 U4/S4 PTX 的实际 SASS lowering，而不是继续把主要精力放在
DRAM、转换 kernel 或 partial 存储上。

### 6.6 为什么不能用隐藏 E0M3 路径替换正式 O3

社区发现的 SM120 E0M3 OMMA 是未公开、需要 CUBIN/SASS patch 的实验路径，而且其
码本为 `-7...7` 并包含正负零，不是标准 two's-complement S4 `-8...7`。它也不能直接
表示 O3 low U4 的 `0...15`，会改变当前 low-U4/high-S4 Split 与 Q4 数值语义。

因此隐藏 E0M3 可以作为独立探索性 variant，但不能用于替换本文的正式 O3，也不能用
它的 FP4 级指令微基准吞吐来声称当前 O3 应当达到目标。

## 7. 结论

| 项目 | O1 | O3 |
|---|---|---|
| 当前核心路径 | 原生 S8×S8 IMMA | PTX U4×S4/S4×S4，SASS lowering 为 INT8 IMMA+位操作 |
| 当前 tile | `128×64×64` | `64×32×128` |
| 数据搬运/调度 | TMA + cooperative warp specialization | TMA + cooperative warp specialization |
| partial | 寄存器 | 寄存器 |
| scale | 软件 K32 | 软件 G128 |
| Compute-only median | 0.623776 ms | 2.065672 ms |
| MSE median vs O0 | `9.8234e-09` | `6.6530e-03` |
| 关键结论 | 当前性能—精度折中较好 | 正确实现 Split 语义，但当前工具链的 SASS 路径限制吞吐 |

O1 已把主要可控开销集中到不可避免的逐 K32 软件 scale；O3 也已经采用 TMA、warp
specialization、寄存器 partial、scale 共享、LDSM、对齐 tile 和循环完全展开。O3 未达到
O1 一半吞吐的根因不是缺少常规 kernel 优化，而是正式 U4/S4 PTX 在 SM120 上实际被
编译为 INT8 IMMA 加大规模位操作。在不改变论文 Split/Q4/G128 实验定义的前提下，
当前证据表明目标无法仅靠继续微调 tile 或 pipeline 达成。

## 8. 相关实现与证据

- O1 正式实现：`csrc/sm120/o1_gemm.cu`
- O3 正式实现：`csrc/sm120/o3_gemm.cu`
- O1 验证：`scripts/validate_o1.py`
- O3 验证：`scripts/validate_o3.py`
- 正式实验配置：`configs/experiment/o0_o1_o2_o3_o4_4096.yaml`
- O3 当前性能边界：`docs/o3_half_o1_optimization_report.md`
- O3/O4 NCU 分析：`docs/o3_o4_ncu_bottleneck_report.md`
- 全部后端最终结果：`docs/o0_o4_final_results_report.md`

服务器上的最终 O3 佐证文件为：

```text
reports/o3_target_half/iteration18_unroll32/o3_full.ncu-rep
reports/o3_target_half/iteration18_unroll32/o3_full_raw.csv
reports/o3_target_half/iteration18_unroll32/o3_full_source.csv
reports/o3_target_half/iteration18_unroll32/audit/summary.txt
reports/o3_mma_lowering/
```

其中 `o3_full.ncu-rep`/CSV 支撑正式 kernel 的动态指标，`audit/summary.txt` 支撑同入口
PTX/SASS 判定，`o3_mma_lowering/` 支撑排除 TMA、scale 和完整 GEMM 调度后的 MMA
lowering 结论。

## 9. 附录：两路原生 INT8 的 O3 反事实诊断

为验证 O3 的主要性能损失是否来自 SM120 对 U4/S4 PTX 的低效 lowering，工程中加入了
一个**不参与正式实验、也不替换 production O3** 的诊断后端：

```text
A_int8 = low_u4 + 16 × high_s4
low_u4  → int8 0...15
high_s4 → int8 -8...7
Q4 W    → int8 -8...7

INT8 MMA(low, W) + 16 × INT8 MMA(high, W)
→ G128 INT32 partial
→ A_scale × W_scale
→ FP32 accumulator/output
```

该路径保持 O3 的 Split、Q4、G128 和 FP32 累加语义，但使用两组
`m16n8k32 S8×S8→S32` Tensor Core 指令，因此不满足正式 O3 的“INT4 Tensor Core”
指令要求。它仅回答一个反事实问题：若避开 U4/S4 lowering，O3 可以快到什么程度。

### 9.1 24 个真实样本结果

测试仍使用 24 个 Llama-2-7B FP16 prefill trace、`M=N=K=4096`、50 次预热、
200 次正式测量和单 CUDA stream。正式 O3 与 INT8×2 在样本和计时模式间交错运行。

| 路径 | Conversion-only total median | Compute-only median | Cold total median | Steady-state total median | MSE vs O0 median | MSE vs O0 mean |
|---|---:|---:|---:|---:|---:|---:|
| 正式 O3 | 0.046124 ms | 2.038248 ms | 2.092336 ms | 2.060576 ms | 0.0066530102 | 0.0075788444 |
| O3 INT8×2 诊断 | 0.062862 ms | 1.422664 ms | 1.502864 ms | 1.465560 ms | 0.0066530102 | 0.0075788444 |

INT8×2 相对正式 O3 的 24 样本配对 compute-only 加速比中位数为 `1.4325×`。
两者的 MSE 在显示精度下相同；4096³ 合成验证中，INT8×2 相对 O3 数学参考的
最大绝对误差为 `9.1553e-5`，MSE 为 `4.1752e-11`。

另一次同进程、同样本、交错顺序的 compute-only O1 对照得到：

| 路径 | 24 样本 latency median | 相对 O1 吞吐 |
|---|---:|---:|
| O1 production | 0.619624 ms | 100% |
| O3 INT8×2 诊断 | 1.415520 ms | 43.79% |

因此该诊断路径已经接近、但尚未达到 O1 的 50% 吞吐目标：其延迟为 O1 的
`2.284×`，比理想的严格 `2×` 下界高约 14.2%。剩余差距来自两路 MMA 之外的
low/high fragment 装载与合并、G128 scale 处理，以及诊断路径把 4-bit 输入展开为
int8 后增加的 shared-memory 数据量。

### 9.2 同入口 SASS 与资源审计

诊断 kernel `adangel_o3_split_int8x2_tma_ws` 的最终审计结果为：

```text
REG=79, STACK=0, LOCAL=0
static IMMA.16832.S8.S8 = 32
TMA/UTMALDG              = 4
U4/S4 IMMA               = 0
LDL/STL                  = 0
```

这证明结果来自同一个 TMA + 两路原生 INT8 IMMA kernel，且没有寄存器 spill。
正式 O3 仍保持原有 production 选择；该诊断后端只能通过内部绑定
`_benchmark_o3_split_int8x2` 调用。

相关文件：

- `csrc/sm120/o3_int8_split_diagnostic.cu`
- `scripts/validate_o3_int8x2.py`
- `scripts/benchmark_o3_int8x2_diagnostic.py`
- `tests/integration/test_sm120_o3_int8x2.py`

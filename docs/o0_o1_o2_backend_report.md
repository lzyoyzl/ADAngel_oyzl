# O0/O1/O2 后端实现、差异与实验测量说明

> 文档定位：面向实验汇报、阶段答辩和结果解读。本文描述 RTX 5090（SM120a）上的当前正式后端，以及已经采用的 MSE 与计时口径；历史 baseline 数据会单独标注，不能与优化后 production 混用。

## 1. 实验目标与公共起点

本实验固定计算：

\[
Y=AW^T,\qquad M=N=K=4096
\]

- \(M=4096\)：激活矩阵行数，即本次 prefill 的有效 token 数；
- \(N=4096\)：Linear 层输出维度；
- \(K=4096\)：Linear 层输入维度，也是点积的归约维度；
- 三种配置均生成 `[4096,4096]` FP32 输出。

三种配置从完全相同的公共准备数据开始：

```text
A_int8      [M,K]      int8
A_scale     [M]        fp32
W_mxfp4     [N,K/2]    packed uint8，每字节存放两个 E2M1
W_scale     [N,K/32]   UE8M0 uint8
```

它们表示的近似实数为：

\[
\hat A_{m,k}=A_{\mathrm{int8},m,k}s^A_m
\]

\[
\hat W_{n,k}=E2M1(W_{n,k})s^W_{n,g},\qquad g=\lfloor k/32\rfloor
\]

从原始 FP16 trace 生成上述公共数据属于所有配置共享的离线准备，不计入 O0/O1/O2 转换时间。实验比较的是：从同一 `A_int8/A_scale/W_mxfp4/W_scale` 出发，使用不同计算路径得到 FP32 输出所需的转换开销、GEMM 时间、端到端时间和相对 O0 的输出误差。

因此，本实验回答的是“完整配置在 RTX 5090 上的系统性能”，不是只比较某一条 Tensor Core 指令的理论峰值。

## 2. 三种后端总体数据流

```mermaid
flowchart LR
    I["公共输入<br/>A_int8 + A_scale<br/>W_mxfp4 + W_scale"]
    I --> O0A["O0-A<br/>INT8 反量化为 FP16"]
    I --> O0W["O0-W<br/>MXFP4 反量化为 FP16"]
    O0A --> O0G["cuBLASLt FP16 Tensor Core GEMM<br/>FP32 accumulate/output"]
    O0W --> O0G
    I --> O1W["O1-W<br/>E2M1 精确映射为 INT8 base"]
    I --> O1G["TMA + cooperative warp specialization<br/>逐 K32 INT8 MMA + 软件 scale"]
    O1W --> O1G
    O1G --> O1Y["FP32 output"]
    I --> O2A["O2-A<br/>反缩放并重量化为 MXFP4<br/>生成 SFA"]
    I --> O2W["O2-W<br/>W_scale 重排为 SFB"]
    O2A --> O2G["CUTLASS SM120<br/>原生 block-scaled MXFP4 MMA"]
    O2W --> O2G
    O2G --> O2Y["FP32 output"]
```

三条路径最核心的区别是 scale 在哪里生效：

| 配置 | scale 的处理位置 |
|---|---|
| O0 | GEMM 前反量化时乘入，GEMM 只看到普通 FP16 A/W |
| O1 | 每个 K32 的 INT8 MMA 得到 INT32 partial 后，由软件乘 scale 并累加 |
| O2 | SFA/SFB 作为原生 block-scaled MMA 输入，由硬件指令在 MMA 内处理 |

## 3. O0：反量化为 FP16 后执行标准 GEMM

### 3.1 数值路径

O0 先将公共量化输入恢复成 FP16：

\[
A^{(0)}_{m,k}=FP16(A_{\mathrm{int8},m,k}s^A_m)
\]

\[
W^{(0)}_{n,k}=FP16(E2M1(W_{n,k})s^W_{n,g})
\]

然后执行：

\[
Y^{(0)}=A^{(0)}(W^{(0)})^T
\]

GEMM 输入为 FP16，乘加使用 FP32 accumulator，最终输出为 FP32。

O0 不是在本轮计时区间内“先量化再反量化”。公共 INT8/MXFP4 量化早已在 `prepare_trace.py` 阶段完成并由三种方案共享；O0 在线工作只是从公共量化输入反量化到 FP16，再执行 FP16 GEMM。

### 3.2 工程实现

- 权重转换：`MXFP4 + UE8M0/K32 -> FP16`；
- 激活转换：`INT8 + per-row FP32 scale -> FP16`；
- GEMM：cuBLASLt；
- 强制算法标志包含 HMMA、FP16 input 和 FP32 accumulator；
- `compute_type = CUBLAS_COMPUTE_32F`；
- `split_k <= 1`；
- workspace、转换目标和输出均在计时前分配。

相关代码：

- `csrc/sm120/conversion.cu`
- `csrc/sm120/o0_gemm.cu`

### 3.3 O0 为什么很快

O0 在 GEMM 前完成所有 scale。GEMM 本身是规则的单次 `4096 x 4096 x 4096` FP16 GEMM，cuBLASLt 可以使用成熟的分块、流水线、数据复用和 Tensor Core 调度策略。GEMM 主循环没有逐 K32 scale 解码、INT32-to-FP32 转换或软件 scale 累加。

O0 的代价是生成完整 FP16 A/W，转换输出体积较大。它曾明显快于历史 O1 baseline；
当前 O1 经寄存器 partial、K64 pipeline、scale 共享和 CTA 消融后已反超 O0，但 O0
仍是结构最规则、最成熟的基线。

## 4. O1：INT8 MMA 与逐 K32 软件 scale

### 4.1 权重的精确 INT8 base 映射

E2M1 的非负数值集合为：

```text
{0, 0.5, 1, 1.5, 2, 3, 4, 6}
```

O1 将其精确映射为：

```text
{0, 1, 2, 3, 4, 6, 8, 12}
```

记映射后的整数为 \(B_{n,k}\)，则：

\[
E2M1(W_{n,k})=B_{n,k}/2
\]

该转换只改变数值表示，不使用 `W_scale`，也不会再引入一次舍入误差。

### 4.2 逐 K32 计算公式

普通 INT8 MMA 接收 INT8 A、INT8 B，并生成 INT32 accumulator；它不接受 UE8M0 scale。O1 必须对每个 K32 显式处理 scale：

\[
P^{(g)}_{m,n}=\sum_{k=32g}^{32g+31}A_{\mathrm{int8},m,k}B_{n,k}
\]

\[
Y^{(1)}_{m,n}=\sum_{g=0}^{K/32-1}FP32(P^{(g)}_{m,n})
\left(s^A_ms^W_{n,g}/2\right)
\]

当 \(K=4096\) 时共有 128 个 K32 分组。“逐 K32”不等于启动 128 个全矩阵 GEMM：当前正式实现已把 128 组放入一个 tiled GEMM kernel 的 K 循环，每个最终输出元素只写全局显存一次。

### 4.3 当前正式 kernel

当前 production 使用
`adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup` kernel：

```text
CTA tile       = 128 x 64 x 64
threads/CTA    = 544
producer warp  = 1
consumer warps = 16
TMA pipeline   = 3 stages
MMA            = CuTe m16n8k32, S8 x S8 -> S32
groups/stage   = 2 independent K32 groups
```

INT32 partial根据CuTe fragment坐标直接在寄存器内匹配行/列scale并累加到FP32；
`shared_partial`实现只作为历史A/B baseline保留。

一个 CTA 最终负责 `Y[128,64]`。每个pipeline stage读取`A_tile[128,64]`和
`W_tile[64,64]`，其中包含两个相邻但数值上独立的K32 group：

1. producer warp 通过 TMA 把下一块 A/W K64 tile 搬入三阶段共享内存流水线；
2. producer对两个group的W scale按CTA/column/group各解码一次，写入stage-local shared；
3. 16 个 consumer warp 按 CuTe `TiledMMA` 的显式 thread/value layout 划分输出；
4. consumer依次对stage内group 0、group 1执行`m16n8k32` INT8 MMA；
5. 每次MMA的INT32 partial分别转FP32并乘自己的
   `A_scale * decode_ue8m0(W_scale) / 2`，不跨group合并scale；
6. 每线程所需的A row scale按两个唯一row在K循环外加载；
7. 64个K64 stage（共128个K32 group）完成后，FP32输出只写全局显存一次。

相关代码：

- `csrc/sm120/conversion.cu`
- `csrc/sm120/o1_gemm.cu`

源码还保留 `adangel_o1_shared_partial_baseline`、上一版`register_64x32`和多种CTA
消融，并包含已因spill取消资格的`adangel_o1_register_partial_128x128`。
公开`run_o1()`只选择上述128×64×64 production。`1.661712 ms`和`1.219312 ms`
分别属于历史shared-partial与上一版production，不能代表当前实现。

### 4.4 历史 shared-partial baseline 的关键瓶颈

历史 baseline 没有全局 partial buffer，但 partial 未全程留在寄存器中：

```text
WMMA INT32 fragment（寄存器）
  -> wmma::store_matrix_sync
  -> shared_storage.shared_partial（共享内存）
  -> consumer named barrier
  -> 按逻辑输出坐标重新读取
  -> INT32 转 FP32、乘 scale、累加到 FP32 寄存器
  -> 第二次 barrier，允许下一组覆盖 partial 区域
```

原因是 `nvcuda::wmma::fragment` 的寄存器元素与逻辑矩阵坐标之间的映射不适合直接依赖。先存为行主序共享内存，可以明确找到每个 `(row,column)` partial 并匹配正确的行、列 scale，但会引入额外共享内存流量和同步。

对 `4096^3` 作量级估算：

- 每 CTA partial tile：`64 x 32 x 4 B = 8 KiB`；
- 每 K32 至少一次写和一次读，约 16 KiB；
- 128 组约为每 CTA 2 MiB；
- 网格共有 `(4096/64) x (4096/32) = 8192` 个 CTA；
- 聚合 partial 共享内存读写约 16 GiB，此外还有每组同步、类型转换和软件 scale。

这解释了历史 O1 baseline 即使已采用 TMA、warp specialization 和单 kernel，仍明显慢于
O0。当前production已消除shared partial读写和named barrier，并通过K64 pipeline、
CTA scale共享、CTA tile消融与A-scale去重进一步降低软件开销，实测已快于O0。普通
INT8 MMA仍缺乏原生block scale输入，因此每个K32的INT32→FP32、缩放与FP32累加无法
消除；这是O1仍明显慢于O2的主要原因。

## 5. O2：原生 SM120 block-scaled MXFP4 GEMM

### 5.1 激活在线重量化

O2 权重已经是 MXFP4，激活需要从公共 INT8 表示重新量化：

```text
A_int8 * A_scale
  -> 每行每 K32 求 amax
  -> 生成 UE8M0 scale
  -> 归一化并按 RNE 量化为 E2M1
  -> 两个 E2M1 packed 到一个 uint8
```

每个 warp 处理一个 K32。对非零块：

\[
e=clamp(\lfloor\log_2(amax)\rfloor-2,-127,127),\quad
s=2^e,\quad scale\_code=e+127
\]

全零块使用 `scale_code=127`，即 scale 为 1。E2M1 量化采用 round-to-nearest、ties-to-even；偶数 K 放低 nibble，奇数 K 放高 nibble。

### 5.2 SFA/SFB scale layout

自然顺序 scale 是 `A_mx_scale[M,K/32]` 和 `W_scale[N,K/32]`。SM120 MXFP4 MMA 对 scale fragment 的物理排布有专门要求，因此 O2 使用 `cutlass::detail::Sm1xxBlockScaledConfig<32>` 生成 SFA/SFB physical layout：

- `A_mx_scale natural -> SFA`；
- `W_scale natural -> SFB`。

O2-W 确实存在转换开销，但它只是 `W_scale` 的布局重排，不是权重重新量化。`W_mxfp4` packed 数据不复制、不转置，直接解释为 ColumnMajor `B[K,N]`。

### 5.3 正式 CUTLASS kernel

O2 使用 CUTLASS `GemmUniversalAdapter`：

```text
A/B element          E2M1 MXFP4
Scale                UE8M0, vector size 32
Accumulator/output   FP32
CTA tile             128 x 128 x 256
Cluster              1 x 1 x 1
Mainloop             TMA
Schedule             KernelTmaWarpSpecializedCooperative
Stage policy         StageCountAutoCarveout
```

目标 MMA：

```text
mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.
m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
```

O2 的 MMA 接口原生接收 MXFP4 数据和 UE8M0 scale。逐 K32 scale 仍然存在，但由 block-scaled MMA 主循环按硬件规定消费，不需要像 O1 一样在每个 K32 后执行软件 scale 路径。

正式运行前调用 `can_implement` 和 `initialize`；条件不满足时直接报错，不允许退化到软件参考实现。相关代码为 `csrc/sm120/o2_cutlass.cu`。

## 6. O0/O1/O2 集中对比

| 项目 | O0 | O1 | O2 |
|---|---|---|---|
| 激活在线处理 | INT8 反量化为 FP16 | 保持 INT8 | 反缩放并重量化为 MXFP4，SFA 重排 |
| 权重在线/冷启动处理 | MXFP4 反量化为 FP16 | E2M1 精确映射为 INT8 base | 只把 W scale 重排为 SFB |
| GEMM 类型 | FP16 x FP16 | INT8 x INT8 | MXFP4 x MXFP4 |
| Tensor Core partial | FP32 | INT32 | FP32 |
| 最终累加/输出 | FP32 / FP32 | 软件 scale 后 FP32 / FP32 | FP32 / FP32 |
| K32 scale 位置 | GEMM 前乘入 | 每 K32 后软件处理 | 原生 block-scaled MMA 内处理 |
| 搬运/调度 | cuBLASLt 内部 | 显式 TMA + cooperative warp specialization | CUTLASS TMA + cooperative warp specialization |
| CTA tile | cuBLASLt 决定 | `64x32x32` | `128x128x256` |
| 全局 partial buffer | 无 | 无 | 无 |
| 最终输出写入 | 一次 | 一次 | 一次 |
| 当前主要优势 | 标准大 GEMM 高度优化 | 保留 INT8 激活；E2M1 映射精确 | 原生 FP4 吞吐高；scale 由 MMA 消费 |
| 当前主要代价 | 生成完整 FP16 A/W | 软件 K32 scale、共享内存 partial 中转 | 激活再次量化；SFA/SFB layout |

三种实现使用不同 CTA tile 是合理的。tile 选择受 MMA shape、数据位宽、共享内存、寄存器压力、scale 布局和流水线深度共同约束。强制使用同一 tile 不会让比较更公平，反而可能让某种数据类型采用明显不适合的实现。

## 7. 比较的公平性与结论边界

共同条件包括：同一 RTX 5090、同一 CUDA stream、同一批 24 个真实 Llama-2-7B FP16 prefill trace、同一公共量化输入、相同 `4096^3` 形状、相同 FP32 输出、相同 warmup/repeat/CV 门限，以及正式后端正确性测试与指令审计。O1/O2 也都采用 TMA、cooperative warp specialization 和单 kernel tiled GEMM。

不可避免的差异来自方案定义：O0 需要 FP16 反量化；O1 需要 E2M1-to-INT8 和软件 K32 scale；O2 需要激活 MXFP4 重量化、SFA/SFB layout 和原生 block-scaled MMA。

因此，实验可以公平回答三种完整配置在当前硬件和默认优化水平下的转换、GEMM、端到端性能及输出差异；但不能把结果解释为 FP16、INT8、MXFP4 单条指令理论吞吐的纯粹对比。尤其 O1-GEMM 包含软件 scale、类型转换、同步和 FP32 累加。

## 8. MSE 的定义、计算与解读

### 8.1 参考输出与公式

每个样本只保存一次 O0 `compute_only` FP32 输出，作为唯一主参考：

\[
Y_{ref}=Y_{O0}
\]

\[
MSE(O_i,O0)=\frac{1}{MN}\sum_{m=1}^{M}\sum_{n=1}^{N}
\left(Y_{O_i,m,n}-Y_{O0,m,n}\right)^2
\]

实现先将两个 FP32 输出转为 FP64，再做差、平方和 mean reduction，以降低指标归约自身的数值误差。输出含 NaN/Inf 时直接判错。

### 8.2 MSE 表示什么

MSE 比较两个完整 `[4096,4096]` 输出矩阵，不是比较权重编码、激活编码或单个 partial。

- O0 对自身 MSE 恒为 0；
- O1 与 O0 共用同一 INT8 激活和 MXFP4 权重，且 E2M1-to-INT8 base 映射精确，差异通常很小；
- O2 对激活执行额外一次 MXFP4 重量化，通常比 O1 有更明显的 MSE；
- MSE 是绝对误差尺度，受输出幅度影响，不等于模型困惑度、任务精度或相对误差。

O0 本身也是从公共量化输入反量化得到的基线，不等于原始 FP16 模型输出。因此 `MSE vs O0` 不能度量公共 INT8/MXFP4 预处理相对原始 FP16 trace 的全部误差。

### 8.3 24 样本汇总

对每个 variant 报告 24 个逐样本 MSE、median、IQR、最大值和 bootstrap median 95% 置信区间。运行器用 `compute_only` 输出计算 MSE，再把同一 variant/sample 的 MSE 写入该样本四种 mode 的记录；不能把四个 mode 中重复的同一个 MSE 当成四次独立观测。

当前一次 24 样本正式运行的摘要为：

| variant | median MSE vs O0 | max MSE vs O0 |
|---|---:|---:|
| O0 | 0 | 0 |
| O1 | `9.8234e-09` | `2.7278e-08` |
| O2 | `3.7960e-03` | `1.5460e-02` |

以上用于说明当前结果量级；最终汇报的精确 IQR 和 bootstrap CI 应从已验收 run 生成的 `04_mse.csv` 读取。

## 9. 四种计时模式

### 9.1 Conversion-only

只回答格式转换本身需要多久：

| 配置 | 转换阶段 |
|---|---|
| O0 | O0-W：MXFP4-to-FP16；O0-A：INT8-to-FP16 |
| O1 | O1-W：MXFP4 E2M1-to-INT8 base |
| O2 | O2-W：W_scale natural-to-SFB；O2-A：反缩放、MXFP4 量化、natural scale-to-SFA |

为了让 API 仍返回有效输出，后端可在 conversion-only 计时区间外执行一次 GEMM；该 GEMM 不计入 conversion-only 时间。

### 9.2 Compute-only

所有当前配置所需转换均提前完成并缓存，只测正式 GEMM：

- O0：预先得到 FP16 A/W；
- O1：预先得到 INT8 W，A 本来就是 INT8；
- O2：预先得到 MXFP4 A、SFA 和 SFB；
- `total` 与 `gemm` 是同一条单次直接路径。

O1-GEMM 仍包含逐 K32 软件 scale，因为这是 O1 计算语义的一部分。

### 9.3 Cold end-to-end

从公共 prepared inputs 出发，包含该配置的全部转换与 GEMM：

```text
O0 cold = W反量化 + A反量化 + FP16 GEMM
O1 cold = W转INT8 + O1 fused tiled GEMM
O2 cold = W_scale重排 + A重量化/SFA重排 + MXFP4 GEMM
```

它表示没有任何配置专用缓存时的一次调用成本。

### 9.4 Steady-state end-to-end

假定静态权重侧转换已经缓存：

```text
O0 steady = A反量化 + FP16 GEMM
O1 steady = O1 fused tiled GEMM
O2 steady = A重量化/SFA重排 + MXFP4 GEMM
```

O1 没有在线激活转换，因为公共激活已经是 O1 所需 INT8；O2 缓存 SFB，但每个新激活仍需量化并生成 SFA。

## 10. 双轨计时方法

当前项目采用“转换阶段批量摊销计时 + 端到端单次直接计时”。

### 10.1 转换组件：批量摊销

O0-W、O0-A、O1-W、O2-W 和 O2-A 都是微秒级操作。单次 CUDA Event 容易被 event 分辨率和偶发调度抖动放大，因此每个外层测量样本执行：

```text
event_start
连续执行 conversion_inner_repeats=100 次同一转换
event_end

单次估计 = (event_end - event_start) / 100
```

`conversion_only/total` 也对该 variant 的完整转换序列执行 100 次后摊销。所有 buffer 在计时前分配，计时区间内没有内存申请或文件 I/O。

### 10.2 端到端 total：单次直接路径

以下 total 不做 inner repeat：

- `compute_only/total`；
- `cold/total`；
- `steady_state/total`。

每个外层样本只执行一次对应真实路径。特别是 cold 只包含一次权重转换，steady-state 继续使用缓存后的静态权重。

### 10.3 为什么组件之和不必等于 total

转换组件与端到端 total 相互隔离、分别测量：转换组件来自独立的 100 次批量区间，cold/steady total 来自一次直接执行，各列最终又分别取 median。因此：

\[
median(W)+median(A)+median(GEMM)\neq median(direct\ total)
\]

是正常现象。延迟分解图说明阶段量级，正式端到端结论必须以 `total` 的 direct measurement 为准。

## 11. 统计量与吞吐指标

正式配置默认：

```text
warmup = 50
repeats = 200
single CUDA stream
conversion_inner_repeats = 100
max_cv_percent = 3.0
```

CUDA Event 在预热前统一预分配；预热同步后立即进入正式测量，二者之间不创建Event、
不申请显存且不做文件 I/O，避免CPU侧准备空档引入GPU动态升频阶段。

对于未锁频RTX 5090上的O1 A/B，CV失败采用有上限的成对定点重试：同一
`(sample, mode)` 的shared和register64必须一起重跑，执行顺序按attempt交替；仅当
两边全部阶段同时低于CV门槛且MSE回归通过时接受。接受规则不读取延迟大小，全部尝试
写入结果JSON，避免单边性能挑选。

O0/O1/O2 在每个 mode 内交错运行，以减小温度和 boost 频率漂移造成的系统偏差。

| 指标 | 含义 |
|---|---|
| `mean_ms` | 200 次延迟算术平均值，对极端离群值较敏感 |
| `median_ms` | 第 50 百分位，主延迟指标；一半样本不大于它，一半不小于它 |
| `p5_ms/p95_ms` | 第 5/95 百分位，观察主要分布范围 |
| `iqr_ms` | Q3-Q1，中间 50% 样本跨度 |
| `cv_percent` | population standard deviation / mean x 100%，衡量相对波动 |

正式主实验仍以 `CV < 3%` 标记稳定记录。O1 后端 A/B 在用户选择不锁频后采用动态Boost
诊断策略：CV、P5/P95、原始离群和retry provenance全部保留，但候选性能判断使用逐样本
配对median及bootstrap 95% CI，不因少量CV离群单独否决。锁频复现可切换为严格策略，
此时CV重新成为硬门槛。任何策略都不能手工删除单个timing sample。

### 11.1 GEMM 等效吞吐

\[
Equivalent\ Throughput=\frac{2MNK}{t_{gemm}}
\]

代码换算为 `equivalent_tflops`。它统一使用 dense GEMM 的 \(2MNK\)，便于比较相同矩阵问题的有效吞吐。

“Equivalent TFLOP/s”不表示 O1 实际执行浮点乘法，也不等于 INT8/MXFP4 指令裸峰值。O1 时间包含软件 scale，O2 时间包含原生 block-scale 主循环。

### 11.2 转换吞吐

\[
Conversion\ GB/s=\frac{logical\ bytes\ moved}{median\ time}
\]

逻辑读写字节：

```text
O0-W = N*K/2 + N*(K/32) + 2*N*K
O0-A = M*K + 4*M + 2*M*K
O1-W = N*K/2 + N*K
O2-W = 2*N*(K/32)
O2-A = M*K + 4*M + M*K/2 + 3*M*(K/32)
```

O2-W 统计自然 W scale 读取与有效 SFB scale 写入。O2-A 的三个 scale 项对应 natural scale 写、natural scale 读和有效 SFA 写。未触碰的 physical-layout padding 不计入逻辑字节。

### 11.3 当前 production 的 compute-only 性能量级

`runs/rtx5090_o1_register_final_dynamic` 的24样本跨样本median摘要如下。该run按用户选择
不锁频，使用动态Boost诊断口径；288条记录中90条触发普通`CV>=3%`标记，因此必须同时
披露`tables/00_stability.csv`，不能称为“全部阶段CV通过”的锁频稳定性结果。

上一版正式run应保留为历史结果：O0/O1/O2的跨样本median分别为
`0.766720/1.219312/0.134624 ms`。其中O1是`register_64x32`，不是当前production。

当前`128x64x64` production在24个真实样本、warmup=50、repeats=200的同进程
配对验收中得到：

| implementation | median of sample medians | equivalent throughput | paired speedup |
|---|---:|---:|---:|
| O0 | `0.758048 ms` | `181.31 TFLOP/s` | `1.000x` |
| 上一版 O1 `register_64x32` | `1.209136 ms` | `113.67 TFLOP/s` | — |
| 当前 O1 `128x64x64` | `0.634232 ms` | `216.70 TFLOP/s` | `1.198x vs O0` |

当前O1相对O0的平均配对加速为`1.1975x`，bootstrap 95% CI为
`[1.1935,1.2024]`；相对上一版O1平均配对加速为`1.9407x`，95% CI为
`[1.9075,1.9823]`。数值与上一版O1在容差内一致，逐样本O1-vs-O0 MSE仍在约
`1e-12`到`3e-8`量级。

production切换后已经重新生成完整288条记录；其正式诊断结果见11.4。任何性能数字
都不是代码中的绝对延迟门槛。

### 11.4 当前 production 的正式24样本诊断结果

当前production的完整run包含24样本、3个variant、4种mode，共288条记录。跨24样本
的median-of-medians如下：

| 口径 | O0 | 当前 O1 | O2 | O1相对O0 |
|---|---:|---:|---:|---:|
| compute-only GEMM | `0.768048 ms` | `0.629416 ms` | `0.136864 ms` | `1.220x`（聚合median之比） |
| cold total | `0.838392 ms` | `0.667272 ms` | `0.197880 ms` | `1.256x` |
| steady-state total | `0.807112 ms` | `0.623248 ms` | `0.208248 ms` | `1.295x` |

compute-only等效吞吐的跨样本median为O0 `178.946 TFLOP/s`、O1
`218.359 TFLOP/s`、O2 `1004.201 TFLOP/s`。O1相对O0的MSE median为
`9.8234e-9`、max为`2.7278e-8`；O2相对O0的MSE median为`0.0037960`、
max为`0.0154600`。逐样本O1/O0 speedup的median为`1.2186x`；它与聚合median
相除得到的`1.220x`口径接近，但两者定义不同。

该run在共享GPU上执行，288条记录中有49条因某个stage的`CV>=3%`被标为不稳定；
运行期间观察到并发外部Python GPU workload，因此这些记录以调度/资源竞争离群解释。
原始记录没有被过滤或由更快的重测结果替换。中位数整体与独立的24样本交错配对结果
一致：后者给出O1相对O0平均配对加速`1.1975x`，95% CI
`[1.1935,1.2024]`。因此本轮结论可作为“大致稳定准确”的共享GPU诊断结果，但不能
宣称全部stage均满足CV门槛或运行期间GPU独占。

## 12. O1 寄存器 partial 候选与 CTA 消融

> **状态：`register_128x64_k64_scale_shared_row_dedup` 已在RTX 5090通过正确性、
> 24样本配对性能和候选TMA+IMMA/无spill审计，现已选为production。相对O0平均
> 加速约`1.198x`，bootstrap 95% CI约为`[1.194,1.202]`。完整runner、正式审计与
> 四表四图均已生成；共享GPU导致的CV离群按11.4诊断口径披露。**

当前实现状态如下：

| 实现 | CTA | producer/consumer | MMA | partial 路径 | 状态 |
|---|---|---|---|---|---|
| `shared_partial` | `64x32x32` | `1/8` | WMMA m16n16k16×2 | shared 中转与两次 barrier | 历史 A/B baseline |
| `register_64x32` | `64x32x32` | `1/8` | CuTe m16n8k32 | 寄存器内缩放/累加 | 上一版production |
| `register_128x64...row_dedup` | `128x64x64` | `1/16` | CuTe m16n8k32×2 | K32独立、寄存器FMA | 当前production |
| `...row_dedup_sparse_scale` | `128x64x64` | `1/16` | 同上 | 稀疏consumer scale load | 慢约1.2%，拒绝 |
| `register_128x128` | `128x128x32` | `1/16` | CuTe m16n8k32 | 寄存器内缩放/累加 | 因spill取消资格 |
| O2 | `128x128x256` | CUTLASS cooperative | MXFP4 block-scaled | 硬件消费 scale | 保持不变 |

production使用`thr_mma.partition_C(identity_tensor)`建立accumulator register与
逻辑`(row,column)`坐标。A row scale在K循环前按唯一row放入寄存器；W column scale
由producer每CTA只解码一次并在stage-local shared中分发；INT32→FP32、scale与FP32
FMA均在寄存器内完成。TMA A/W三阶段shared-memory pipeline仍然保留。

晋升证据包括：reference与边界shape正确；24样本O1-vs-O0 MSE相对baseline满足
`rtol=1e-5, atol=1e-12`；compute-only配对加速比bootstrap 95% CI下界>1；同一
候选SASS内同时出现TMA/IMMA且无`LDL/STL` spill。未锁频运行中的CV保留为诊断项，
不设置绝对延迟门槛。

`register_128x128` 只令 M/N tile 与 O2 相同，K tile 仍为32；O2 K tile 为256且在
原生 block-scaled MMA 中消费8组 K32 scale。因此不强制 O1/O2 完整 CTA 相同。


### 12.1 已实施的优化路径

保留当前 O1 已具备的设计：

- TMA 搬运 A/W；
- cooperative warp specialization；
- 单 kernel tiled GEMM；
- 三阶段 A/W shared-memory pipeline；
- 精确 E2M1-to-INT8 base 映射；
- 每 K32 独立 `W_scale`；
- INT32 partial、FP32 最终累加和 FP32 输出；
- 每个输出元素只写全局显存一次。

第一阶段把partial后处理改为：

```text
baseline：
INT32 MMA register fragment
  -> shared partial
  -> barrier
  -> shared reload/remap
  -> scale
  -> FP32 accumulator register

register production：
INT32 MMA result registers
  -> 直接匹配本线程所拥有元素的 row/column scale
  -> INT32-to-FP32 + scale
  -> FP32 accumulator registers
```

“partial留在寄存器”不表示完全不使用共享内存。TMA搬入的A/W tile仍需要共享内存
多阶段流水线；消除的是`shared_storage.shared_partial`及其named barrier。

随后依次实施并消融：

1. W scale由每个consumer warp重复解码改为producer每CTA/column/group解码一次；
2. 用`__fmaf_rn`保持固定的FP32累加顺序；
3. pipeline粒度由K32改为K64，一个stage承载两个数值独立的K32 group；
4. 比较`64x32`、`128x32`、`64x64`、`128x64`和`128x128`输出tile；
5. 根据CuTe fragment坐标把每线程A scale加载从16次去重到2个唯一row；
6. 尝试只由warp-N拥有的lane读取W scale，但该候选慢约1.2%，未晋升。

### 12.2 为什么不能只删除 `store_matrix_sync`

每个 partial 元素需要乘：

\[
s^A_{row}s^W_{column,group}/2
\]

因此必须知道每个 accumulator register 对应 tile 的哪一行、哪一列。`nvcuda::wmma::fragment` 内部元素映射不应作为稳定、公开的逻辑布局直接依赖，当前实现才使用 shared-memory row-major 中转恢复坐标。

当前production使用 CuTe 明确表达寄存器布局：

1. `partition_C(identity_tensor)` 建立 accumulator register 的行列坐标；
2. m16n8k32 atom 一次完成一个 K32 INT8 MMA；
3. A row scale按唯一row预取，W column/group scale由producer解码到stage-local shared；
4. 在寄存器内完成 INT32→FP32、缩放和按 group 顺序累加。

### 12.3 实测收益与仍然存在的成本

已消除：

- 约 16 GiB 量级的聚合 shared partial 读写；
- 每 K32 围绕 partial 重分发的两次 consumer barrier；
- shared partial 容量占用。

仍无法消除：

- 128 个 K32 迭代；
- 每 K32 的 `W_scale` 获取/解码；
- INT32-to-FP32 转换；
- `A_scale * W_scale / 2`；
- 每 K32 对 FP32 accumulator 的乘加；
- 相比原生 block-scaled MMA 更复杂的软件数据通路。

实测当前production相对上一版O1平均加速约`1.94x`，相对O0平均加速约`1.20x`。
它仍约为O2延迟的5倍，因为O2由原生block-scaled MMA消费scale，而O1仍执行大量
INT32→FP32与软件FMA。这里不设置绝对延迟作为正确性门槛。

### 12.4 晋升 production 的验收状态

已经完成：

1. 小规模 Python reference 对齐和边界 shape 测试；
2. `4096^3` compute-only 24样本候选配对与CV定向重测；
3. O1 相对 O0 的 MSE 回归，确认未改变 K32 scale 与累加语义；
4. 生产 kernel 指令审计，确认同一 kernel 中仍有 TMA 和 IMMA；
5. SASS/源码检查，确认不再有 partial shared-memory store/reload；
6. 与上一版 O1 的 compute-only 配对对照。

production常量切换后的四种mode正式runner、完整O0/O1/O2审计与四表四图均已完成。
当前图表使用diagnostic稳定性口径，保留所有CV失败记录与离群原因；若未来取得独占GPU
窗口，可再补充strict复现实验，但不改变候选晋升结论。

候选结果元数据必须包含：

```text
partial_storage = register
shared_partial_redistribution = false
cta_tile = [128,64,64]
groups_per_pipeline_stage = 2
row_scale_loads_per_thread = 2
```

旧图表必须标注为旧O1结果。当前正式run的production元数据已经明确报告
`register_128x64_k64_scale_shared_row_dedup`，可作为当前O1诊断结果。

## 13. 指令审计与结果边界

一次性审计用于证明正式后端没有退化：

- O0：使用 FP16 Tensor Core/HMMA；
- O1：TMA load 与 INT8 IMMA 属于同一个正式 O1 kernel；
- O2：TMA load 与 MXFP4 block-scaled MMA 属于同一个正式 CUTLASS O2 kernel；
- probe/microkernel 只能用于布局和 ISA 验证，不能代替正式 kernel 命中。

审计是正确性与实现真实性门槛，不作为主性能结果，也不等价于完整 GPU profiling。

## 14. 汇报建议结论顺序

1. 三种配置从完全相同的公共量化输入出发，最终输出统一为 FP32；
2. O0 把 scale 前置到 FP16 反量化，随后执行高度优化的标准 GEMM；
3. O1 使用 INT8 Tensor Core，但普通 INT8 MMA 不支持 UE8M0 block scale，必须逐 K32 软件缩放；
4. 当前O1已将partial留在寄存器，并通过K64 pipeline、CTA scale共享、128×64输出tile
   与A row scale去重，在24样本配对测试中快于O0约1.20倍；
5. O2 使用原生 block-scaled MXFP4 MMA，逐 K32 scale 由指令消费，GEMM 性能最高；
6. O1 相对 O0 的 MSE 极小，O2 因激活再次量化而有更明显误差；
7. 转换开销使用批量摊销，端到端 total 使用单次直接计时，不能简单把阶段 median 相加；
8. 最终结论要同时呈现转换成本、compute-only、cold/steady-state 和 MSE，不能只看吞吐图。

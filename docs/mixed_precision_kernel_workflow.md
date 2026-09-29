# A100 O5–O10：实现与验证流程

本文使用当前命名：O5/O6 是 FP16 基线，O7/O8 是双 INT4，O9/O10 是 Binary。
实验定义和运行命令见 [分组协议](o5_o10_a100_protocol.md)。O0–O4 的原有入口保留。

## 1. 共用输入，分开计算路径

每个样本从原始 FP16 trace 生成两套源数据，随后组内三个后端共用同一份编码张量。
源量化只做一次，不计入在线转换时间；不从旧 INT8/MXFP4 二次量化作为正式输入。

| 组 | 共用权重 / 激活 | FP16 基线 | 双 INT4 | Binary |
|---|---|---|---|---|
| A | NVFP4-G128 / MXFP8 E4M3-G128 | O5 | O7 | O9 |
| B | HiF4-G128 / 实验 NV-style FP6 E2M3-G128 | O6 | O8 | O10 |

G128 是沿 K 每 128 个元素共用尺度。这些 G128 扩展和带两级 scale 的 FP6
都按已确认的实验定义实现，不作为 NVIDIA 标准格式宣称。

## 2. 转换器输出什么

令源值为 `v × s`，定点转换为 `q = clamp(RNE(v × 2^F))`，有效尺度为 `s × 2^(-F)`。
HiF4 的内部微指数先并入 v。舍入改变数值；后续补码拆分和 packing 不再改变 q。

| 来源 | 整数 | F | 有效尺度 |
|---|---|---:|---|
| NVFP4 E2M1 权重 | Q4 | 0 | E4M3 block scale × FP32 tensor scale |
| MXFP8 E4M3 激活 | Q8 | −2 | UE8M0 scale × 4 |
| HiF4 合并微指数后的权重 | Q4 | 0 | E6M2 scale |
| FP6 E2M3 激活 | Q6 | 2 | E4M3 block scale × FP32 tensor scale ÷ 4 |

O5/O6 直接解码源值并 RNE 到 FP16，**不经过上述定点舍入**。
O7/O8 将源解码、定点转换、双 nibble packing 和有效 scale 输出融合。
O9/O10 将源解码、定点转换、warp ballot 和 scale 输出融合，不先生成完整 INT8 中间矩阵。
源解码用精确指数位重编码替代通用 `ldexp`；合法编码均另与独立参考核对。

### 物理存储

R 表示矩阵行数，G=K/128。

| 张量 | 实际布局 |
|---|---|
| NVFP4 源 payload | uint8[R,K/2]；偶数 K 在低 nibble |
| MXFP8 源 payload | uint8[R,K] |
| HiF4 源 payload / 微指数 | uint8[R,K/2]；micro8[R,G,2]、micro4[R,G,4] 按 bit packing |
| FP6 源 payload | uint8[R,K]，仅低 6 位有效；不是紧凑 6-bit 存储 |
| O7/O8 权重 | S4 packed uint8[N,K/2] |
| O7/O8 激活 | uint8[2M,K/2]，前 M 行 low U4、后 M 行 high S4 |
| O9 权重 / 激活 | int32[4,N,K/32] / int32[8,M,K/32] |
| O10 权重 / 激活 | int32[4,N,K/32] / int32[6,M,K/32] |

Binary 每个 32-bit word 装同一行、同一平面的 32 个连续 K 元素；第 k 个 lane
对应 word 的第 k 个 bit。int32 是存储容器，计算操作数是 B1，不是 INT32 GEMM。
O10 真正使用六个激活平面，没有补到八平面计算。

有效 FP32 scale 的逻辑 shape 为 [R,G]。INT4 选定配置直接输出物理 [G,R]，
以 stride=[1,R] 的 view 保留逻辑 shape；Binary 选定配置采用自然 [R,G]。
布局选择包含在转换计时中，没有未计时的额外重排。

## 3. Tile 内如何计算

O5/O6 复用 O0 的 cuBLASLt FP16 Tensor Core 选择策略，FP32 累加/输出，不启用 split-K。
算法查询、workspace、输出分配都在计时前。

O7/O8 与 O9/O10 的选定 CTA 均为 **64×128×256**，256 threads，两个 shared-memory stage，
用 `cp.async` 重叠搬运与计算。A100 不使用 TMA。一次 K256 搬运包含两个 G128，
它们必须分别缩放，不能把两个整数 partial 相加后只乘一个 scale。

```text
每个 CTA 持有一个 64×128 的 FP32 输出 tile
for 每个 K256 stage:
    等待当前 shared stage，预取下一 stage
    for 其中两个 G128，按 K 升序:
        在寄存器内计算该 G128 的整数点积
        scale = round_fp32(A_scale[row,g] × W_scale[col,g])
        accumulator = fma(float(integer_dot), scale, accumulator)
最后每个输出元素写回一次
```

**双 INT4：** `a = low_u4 + 16 × high_s4`，先得到两路点积，再在 INT32 寄存器内重构。
Q6 激活先符号扩展为 INT8，所以 O8 仍是两路 INT4，不会自动减少为 O7 的 6/8 工作量。
同一正式函数中必须出现 U4×S4 和 S4×S4 IMMA。沿用 O3 的窄 N-slice fragment
复用、寄存器 partial 和最终向量写回；不把逐组完整输出写入显存。

**Binary：** 权重补码位权为 [1,2,4,−8]；激活为 Q8 的 [1,…,64,−128] 或 Q6 的
[1,…,16,−32]。`m16n8k128.and.popc` 产生各平面对的非负 popcount，按两侧位权乘积
重构有符号整数点积。O9 每个 atom/G128 有 32 个平面对，O10 有 24 个。
权重 fragments 缓存在寄存器并跨激活平面复用，使用两条整数累加链；partial 不落全局内存。
保留相同 G128 顺序和 FP32 FMA 顺序，因此应分别与 O7/O8 输出逐位相同。

## 4. 计时边界

| 模式 | 计时内容 |
|---|---|
| conversion-only | 分别测 W/A 源格式→执行格式；每个 Event 重复 inner 次后除以 inner |
| compute-only | W/A 已转换，仅测一次 GEMM |
| cold | 一次 W 转换 + A 转换 + GEMM 的直接 Event 区间 |
| steady-state | 已缓存 W 转换，一次 A 转换 + GEMM 的直接 Event 区间 |

转换 total 是逐次 W/A 摊销样本之和，再计算统计量；不是联合 Event。
端到端 total 独立实测，不使用各阶段 median 相加。所有 Event、buffer 和计划均预先分配。
源量化、数据 I/O、正确性检查和 MSE reduction 均不计入以上区间。

## 5. 验证与证据

1. 标量参考和 CUDA 转换对照：有限编码、tie、正负零、scale 边界、非法 code。
2. 解包实际 GPU bitplane，与定点 q 逐元素比较；FP16 baseline 与源解码后的 half 比较。
3. 整数 GEMM 与独立逐 G128 参考比较，Binary 与双 INT4 输出逐位比较。
4. 对同一实际函数审计 PTX/SASS：INT4 IMMA 或 Binary BMMA，加 cp.async/LDGSTS；不由 probe 代替。
5. memcheck/racecheck 检查边界与 stage 生命周期。cuBLAS kernel 单独由 NCU 确认 HMMA。
6. 24 个原始 FP16 样本做四模式同输入配对；保留原始计时、CV、GPU 状态和代码/二进制 hash。

主 MSE 是 O7/O9 对 O5、O8/O10 对 O6 的 FP32 输出误差，用 FP64 reduction。
相对旧 O0 的 MSE 仅为辅助列；不是模型下游任务精度。
原始源量化损失、转定点新增损失、kernel 实现误差分别记录，不能相加当作总误差。

实现入口：`csrc/sm80/mixed_conversion.cuh`、`mixed_bitplane.cuh`、`mixed_benchmark.cuh`；
真实实验入口：`scripts/benchmark_a100_mixed_trace.py`。新实验通过专用入口运行，
不把历史 O5/O6 日志静默解释为当前 FP16 基线，也不改变 SM120 的默认后端。

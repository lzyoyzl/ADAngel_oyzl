# O3/O4 Split 与 Bitwise 后端实现报告

## 1. 实验目标与边界

O3/O4 在 RTX 5090（SM120a）上计算：

```text
Y = A @ W.T, M=N=K=4096, FP32 accumulate/output
```

两者都从公共 prepared 输入开始：逐行对称量化的 `A_int8/A_scale`，以及按 G128
量化的 `W_mxfp4_g128/W_scale_g128`。公共 FP16 trace 的初始准备不计入在线开销；
O3/O4 各自需要的 Split、Q4 和 bitplane 转换按四种 timing mode 计量。

本实现对齐论文的 Split/Bitwise 算术，并复用 O1 已验证的工程原则：TMA、cooperative
warp specialization、单 kernel tiled GEMM、寄存器 partial、分组 scale 后 FP32
累加、最终一次输出写回。没有加入 Stream-K、tile autotune 或额外极限优化。

## 2. 公共 G128 MXFP4→Q4 映射

权重先按 128 个 K 元素共享一个 UE8M0 scale 量化为 E2M1。O3/O4 都保留该 G128
scale，只把 scale 内的 E2M1 private value 映射为 Q=4、F=0 的 signed Q4：

```text
E2M1 magnitude: 0, 0.5, 1, 1.5, 2, 3, 4, 6
Q4 magnitude:   0, 0,   1, 2,   2, 3, 4, 6
```

映射使用 round-to-nearest, ties-to-even；符号对称保留。Q4 两个元素打包到一个
byte，偶数 K 位于 low nibble，奇数 K 位于 high nibble。运行时的整数点积结果乘：

```text
A_scale[row] * decode_ue8m0(W_scale_g128[column, group])
```

注意 G128 scale 与原 O0/O1/O2 的 K32 权重是两套公共表示；O3/O4 之间共享同一套
G128 权重，因此两者的数值差异来自计算分解方式，而不是不同的权重量化输入。

## 3. O3：论文 Split

### 3.1 A8 拆分

对任意 signed INT8 `a`，先按原始 two's-complement 位模式解释为 `uint8`：

```cpp
uint8_t u = static_cast<uint8_t>(a);
low_u4  = u & 0x0f;
high_s4 = sign_extend_4bit(u >> 4);
```

恒等式为：

```text
a = low_u4 + 16 * high_s4
```

例如 `a=-3` 的原始位是 `0xfd`，所以 `low=13`、`high=-1`，重构为
`13 + 16*(-1) = -3`。GPU conversion 将 low/high 分别打包为 U4/S4 tensor。

### 3.2 每个 G128 的计算

对第 g 个 G128：

```text
P_low  = MMA_U4xS4(A_low[g],  W_q4[g])
P_high = MMA_S4xS4(A_high[g], W_q4[g])
P_g    = P_low + 16 * P_high
Y     += FP32(P_g) * A_scale * W_scale_g128[g]
```

一个 G128 由两个 K64 MMA subgroup 完成；低/高两路 INT32 partial 都保留在寄存器。
每组 scale 在该组整数 partial 重构后立即应用，不能将不同 G128 partial 先相加再统一
缩放。

### 3.3 Tile 与 pipeline

```text
CTA tile          128 x 16 x 128
producer warps    1
consumer warps    16
pipeline stages   3
MMA               m16n8k64 U4xS4 + S4xS4
```

16 个 consumer warp 排列为 8 个 M warp tile × 2 个 N warp tile。每个 warp 覆盖
`16x8` 输出，因此合计恰好覆盖 `128x16`，无重复、无缺口。producer warp 用 TMA
把 low/high A tile 和 Q4 W tile 搬到三阶段 shared-memory pipeline，并为每个
column/group 解码一次 W scale。

subbyte operand 的寄存器装载不依赖猜测的 `ldmatrix` 组合，而是直接按 pinned CUTLASS
中 `SM80_16x8x64` MMA trait 推导 lane→A/B/C 寄存器坐标。正式 PTX/SASS 仍执行
目标 INT4 MMA；shared memory 只承担 TMA staging，不承担 partial 中转。

## 4. O4：论文 Bitwise

### 4.1 Two's-complement bitplane

激活 A8 拆成 8 个 bitplane，系数为：

```text
[1, 2, 4, 8, 16, 32, 64, -128]
```

Q4 权重拆成 4 个 bitplane，系数为：

```text
[1, 2, 4, -8]
```

因此：

```text
a = sum_i alpha[i] * A_bit[i]
w = sum_j beta[j]  * W_bit[j]
```

这里显式包含 two's-complement 符号 plane，不使用“只统计幅值、符号额外处理”的
近似，也不采用 `7*(Q-1)=21` 的另一种分解。

### 4.2 每个 G128 的计算

每一对 activation/weight plane 执行一次 B1 AND-POPC BMMA：

```text
P_ij = BMMA_AND_POPC(A_bit[i], W_bit[j])
P_g  = sum_{i=0..7} sum_{j=0..3} alpha[i] * beta[j] * P_ij
Y   += FP32(P_g) * A_scale * W_scale_g128[g]
```

所以每个 G128 有 `8*4=32` 个逻辑 BMMA。所有 plane-pair 结果先在 INT32 寄存器中
重构为该组整数 partial，再应用 G128 scale 和 FP32 FMA。

### 4.3 Tile 与 pipeline

```text
CTA tile          128 x 16 x 128
producer warps    1
consumer warps    16
pipeline stages   3
MMA               m16n8k128 B1 AND-POPC
```

warp 排列、一次输出写回和 O3 相同。A/W bitplane 按 32 bit 打包；TMA 每 stage
搬入完整 G128 的 8 个 A plane 和 4 个 W plane。lane mapping 直接来自 pinned CuTe
B1 MMA trait，避免 bit、lane、row/column 或转置错误。

当前实现为了回答“bitplane 转换开销是多少”，在 GEMM 前用独立 GPU kernel 生成
bitplane，并在 conversion-only/cold/steady-state 中显式计量。这与论文 Bitwise
算术一致，但不宣称已经实现论文中的 selective fusion。若未来加入 selective fusion，
必须作为新实现单独命名和 A/B，不能覆盖当前可测量 baseline。

## 5. 与 O1 的关系

共同部分：

- TMA + cooperative warp specialization；
- 三阶段 shared-memory operand pipeline；
- 单 kernel tiled GEMM；
- partial 保留在寄存器；
- 按分组 scale 后做 FP32 累加；
- 最终每个输出元素只写一次；
- 相同四种 timing mode、24 样本和 MSE 口径。

不可避免的差别：

| 项目 | O1 | O3 | O4 |
|---|---|---|---|
| 权重组 | K32 | G128 | G128 |
| 权重整数形式 | `2*E2M1` INT8 | RNE Q4 | Q4 的 4 planes |
| 激活形式 | INT8 原样 | U4 low + S4 high | 8 planes |
| Tensor Core | INT8 IMMA | 两类 INT4 IMMA | B1 AND-POPC BMMA |
| 每组核心分解 | 1 次 K32 MMA | 2 路×2 个 K64 | 32 plane pairs |
| scale | 软件 K32 | 软件 G128 | 软件 G128 |
| CTA | 128x64x64 | 128x16x128 | 128x16x128 |

CTA 不强制相同。不同 MMA atom、每组工作量和自然 warp coverage 决定了不同的合理
tile；强行统一 CTA 会改变资源压力或产生未覆盖输出，反而降低可比性。

## 6. 计时口径

| mode | O3 | O4 |
|---|---|---|
| conversion-only | W: MXFP4→Q4；A: Split | W: MXFP4→Q4→4 planes；A: 8 planes |
| compute-only | 预转换，只测 INT4 GEMM | 预转换，只测 BMMA GEMM |
| cold | W 转换 + A 转换 + GEMM | W 转换 + A 转换 + GEMM |
| steady-state | 缓存 Q4 W，只保留 A Split + GEMM | 缓存 Q4/W planes，只保留 A planes + GEMM |

转换阶段采用 CUDA Event 批量执行 `conversion_inner_repeats=100` 后除以 100；
compute-only/cold/steady-state 的端到端 total 采用单次直接计时。所有 buffer、TMA
descriptor 和静态转换缓存都在计时边界外预分配。

## 7. MSE

O0 的 FP32 输出是唯一主参考：

```text
MSE(Oi,O0) = mean((Y_Oi - Y_O0)^2), i in {1,2,3,4}
```

两侧输出转 FP64 后做 reduction。O3/O4 参考不是用于最终 MSE 的替代基线；它们只验证
kernel 是否实现预定的 Split/Bitwise 数学语义。最终报告包含逐样本 MSE，以及 24 样本
median、IQR、max 和 bootstrap 95% CI。

## 8. 已通过的后端验收

RTX 5090、CUDA 12.8、CUTLASS 4.5.2 pinned commit 上：

| 验收 | O3 | O4 |
|---|---:|---:|
| 128³ max abs error vs semantic reference | 0 | 0 |
| 4096³ max abs error vs semantic reference | 9.1553e-05 | 9.1553e-05 |
| 4096³ smoke compute median | 2.1455 ms | 8.9760 ms |
| 4096³ smoke compute CV | 0.121% | 0.354% |
| production resource | REG 55, STACK 0, LOCAL 0 | REG 55, STACK 0, LOCAL 0 |
| same-entry instruction audit | TMA + U4/S4/S4/S4 MMA | TMA + B1 AND-POPC BMMA |

这些是后端验收 smoke 数据，不替代 24 个真实样本、warmup=50、repeats=200 的正式
实验。O4 比 O3 慢不表示错误；它每个 G128 必须执行 32 个 plane-pair BMMA，而 O3
只执行低/高两路 INT4 分解。

## 9. 复现命令

```bash
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation --no-deps
python scripts/validate_o3.py
python scripts/validate_o4.py
python -m pytest tests/integration/test_sm120_o3_o4.py -q --run-sm120

python scripts/validate_o3.py --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 | tee reports/o3_4096_validation.json
python scripts/validate_o4.py --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 | tee reports/o4_4096_validation.json

EXTENSION_DIR=$(python -c "import torch,pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit_o34
python -m adangel doctor --require-native
```

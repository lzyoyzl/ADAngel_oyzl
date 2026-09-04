# O3 性能优化与“O1 一半吞吐”目标结论

## 1. 结论摘要

本轮在 RTX 5090（SM120a）、CUDA 12.8 和固定 CUTLASS 4.5.2 commit 上完成了 O3
的实现—审计—正确性—MSE—性能—NCU 迭代。

最终 production 为：

```text
implementation  m64_n32_k128_aligned_factor_16w
CTA tile        64 x 32 x 128
pipeline        2-stage TMA，1 G128/stage
warps           1 producer + 16 consumers
MMA semantics   U4xS4 low + S4xS4 high，m16n8k64
partial         INT32 registers
acc/output      FP32 registers / FP32
```

它在24个 Llama-2-7B真实样本上相对上一版 O3 获得稳定的 `1.0313x` compute-only
配对加速，正确性、MSE、CV 和无 spill 审计全部通过，因而已晋升。

但目标没有达到。当前 O1 compute-only 为 `0.622136 ms`，一半吞吐对应 O3 延迟上限
`1.244272 ms`；新 O3 正式 `4096³` 延迟为 `2.037664 ms`，只达到 O1 约30.5%的
等效吞吐。

原因不是 DRAM、TMA 或 conversion，而是 SM120 对旧式 sub-byte integer PTX 的物理
lowering：O3 的 PTX 保留两路 U4/S4 语义，SASS 却由 U8×S8/S8×S8 IMMA 加大量
`LOP3/SHF/IMAD` 实现。RTX 5090 没有可供本实验调用、并提供2倍 INT8吞吐的独立原生
INT4 SASS路径。

## 2. 必须保持的实验约束

本轮所有可晋升候选均保持：

- `A8 = A_low_u4 + 16*A_high_s4`；
- 权重为 MXFP4-G128 经 RNE 映射的 Q4；
- 每个 G128 独立应用 UE8M0 scale；
- low U4×S4 与 high S4×S4 两路 sub-byte PTX MMA；
- TMA + cooperative warp specialization；
- 单 kernel tiled GEMM；
- INT32 partial 和 FP32 accumulator 留在寄存器；
- 每个输出元素只写一次；
- FP32 最终输出；
- conversion-only、compute-only、cold、steady-state 和 MSE 口径不变。

biased-high U4 与显式 INT8 重写只作为诊断，不属于正式 O3，因为它们不再保留上述
signed-high 两路执行约束。

## 3. 优化前的瓶颈

上一版 production 为 `n16_k128_cute_ldsm`：

```text
CTA             128 x 16 x 128
consumer layout 8 M warps x 2 N warps
pipeline        2-stage TMA
registers       55/thread
```

定向 NCU 表明：

| 指标 | 上一版 O3 |
|---|---:|
| NCU duration | 2.073088 ms |
| 动态 SASS | 约 2.798B |
| SM instruction throughput | 约 92.84% |
| IMMA pipe active | 约 13.55% |
| active warps/scheduler | 8.38 |
| eligible warps/scheduler | 3.65 |
| math-pipe throttle | 3.93 cycles/issue |

DRAM吞吐较低，TMA等待不是主导；大量动态指令用于 packed4 operand 的拆位、符号
扩展、lane排列和整数地址/索引。真正的 MMA 只占执行工作的一小部分。

## 4. 本轮候选与最终选择

本轮测试了以下方向：

1. 每 warp 处理两个 M atom，复用 B fragment；
2. W scale 由少数 lane 加载再 warp broadcast；
3. row A scale 移出全部 G128循环；
4. 对齐的4096³快路径，删除 M/N tail判断；
5. `64x32`、`32x64` 平衡 CTA；
6. 每个 pipeline stage 搬运两个 G128；
7. 将平衡 CTA、对齐快路径和 row-scale factoring 组合。

最终只有第7项形成稳定且统计显著的收益。新实现的关键变化是：

- CTA 从 `128x16x128` 改为 `64x32x128`，仍是16个 consumer warp，每 warp只负责
  一个 `16x8` atom；
- producer 仍对每个 CTA/column/group 只解码一次 W scale并放到 stage-local shared；
- A row scale 每线程只加载一次，并在完成全部 G128 FP32 accumulation 后统一乘；
- 正式 `4096³` 删除不需要的 tile边界判断；
- 非整 `64x32` 输入自动回退到上一版 predicated kernel，公开 API 仍支持边界尺寸。

双 G128 pipeline、额外 fragment复用、warp scale broadcast 和独立 MMA chain 均未带来
收益：它们增加寄存器活跃范围、shuffle、同步或指令调度压力，而没有减少 ptxas 固定的
sub-byte lowering 工作。

## 5. 24 个真实样本配对结果

实验使用相同进程、相同输入、交替运行顺序，warmup=50、repeats=200、
conversion_inner_repeats=100；24个样本均满足各 stage `CV<3%`。

| 指标 | 上一版 O3 | 新 production |
|---|---:|---:|
| compute-only 跨样本 median | 2.152408 ms | 2.076480 ms |
| cold total 跨样本 median | 2.210080 ms | 2.133232 ms |
| steady-state total 跨样本 median | 2.176184 ms | 2.101496 ms |
| MSE vs O0 median | 0.006653010193 | 0.006653010195 |
| MSE vs O0 mean | 0.007578844398 | 0.007578844400 |
| MSE vs O0 max | 0.018079114690 | 0.018079114689 |

| 配对统计 | compute-only | cold total | steady total |
|---|---:|---:|---:|
| speedup几何均值 | 1.031277x | 1.030847x | 1.031777x |
| speedup median | 1.031131x | 1.030656x | 1.033178x |
| bootstrap mean 95% CI | [1.028552, 1.033760] | [1.028305, 1.033215] | [1.029290, 1.033816] |

MSE 的末位差来自 row scale 与 FP32 accumulator 的等价结合顺序，远低于既定回归门槛；
没有引入新的量化误差。

晋升后又通过公开 production调度独立重跑24样本，得到72条记录且全部 stage
`CV<3%`、全部 `production_selected=true`：compute-only、cold和steady-state的
跨样本 median 分别为 `2.054640 ms`、`2.120000 ms` 和 `2.078960 ms`；MSE median/
mean 分别为 `0.006653010195` / `0.007578844400`。这次独立绝对值与上表配对轮次的
轻微差异来自未锁频 GeForce的动态频率；晋升判断仍以上表同进程配对速度比为准。

## 6. 新 production 的正式 4096³ 结果

重新编译后使用公开 `production` 调度，warmup=50、repeats=200：

### 6.1 Conversion-only

| 阶段 | median ms | mean ms | CV |
|---|---:|---:|---:|
| W: MXFP4-G128→Q4 | 0.029934 | 0.029934 | 0.066% |
| A: INT8→low U4/high S4 | 0.016243 | 0.016239 | 0.047% |
| isolated conversion total | 0.046176 | 0.046173 | 0.045% |

### 6.2 Compute-only

| 阶段 | median ms | mean ms | CV |
|---|---:|---:|---:|
| O3 GEMM | 2.037664 | 2.031853 | 0.717% |

### 6.3 Cold

| 阶段 | median ms | mean ms | CV |
|---|---:|---:|---:|
| direct end-to-end total | 2.105504 | 2.104864 | 0.117% |

### 6.4 Steady-state

| 阶段 | median ms | mean ms | CV |
|---|---:|---:|---:|
| direct end-to-end total | 2.068704 | 2.062792 | 0.671% |

`max_abs_error vs semantic reference = 9.1552734375e-05`；输出是 finite FP32。

## 7. NCU 与指令审计

新旧 production 的 targeted NCU 对比如下。NCU replay时间只用于定位瓶颈，正式性能
仍以上一节 CUDA Event为准。

| 指标 | 上一版 | 新 production | 变化 |
|---|---:|---:|---:|
| NCU duration | 2.073088 ms | 1.989248 ms | -4.0% |
| 动态 SASS | 2.798B | 2.559B | -8.5% |
| instruction throughput | 92.84% | 94.73% | 更接近饱和 |
| IMMA pipe active | 13.55% | 14.11% | 略升 |
| math-pipe throttle | 3.93 | 4.18 | 仍高 |
| eligible warps/scheduler | 3.65 | 3.56 | 基本不变 |

production审计结果：

```text
PTX same entry   TMA + U4xS4 MMA + S4xS4 MMA
SASS             U8xS8/S8xS8 IMMA + bit operations
resources        REG=55, STACK=0, LOCAL=0
spill            none
```

这解释了“输入是4 bit，为何没有接近2倍 INT8吞吐”：PTX 是虚拟 ISA接口，最终执行
能力由 SASS和硬件决定。SM120 的原生整数矩阵路径是 INT8；ptxas 必须把 packed4
语义展开成多条 INT8 IMMA及位操作。扩大 tile 只能摊薄部分地址/调度开销，不能删除
这组物理工作。

## 8. 目标判定与后续边界

| 项目 | 数值 |
|---|---:|
| O1 compute-only | 0.622136 ms |
| O3 一半 O1吞吐的延迟上限 | 1.244272 ms |
| 新 O3 compute-only | 2.037664 ms |
| 新 O3 / O1等效吞吐 | 30.5% |
| 是否达到50% | 否 |

继续进行一般 CTA、stage或scale微调，无法再提供所需的约1.64倍加速。两个技术方向
可能接近目标，但都超出当前正式实验定义：

1. 在具有真正原生 INT4矩阵指令的硬件/ISA上复现同一 O3；
2. 把 Split代数重写成原生 INT8 GEMM与修正项，另立新的 variant，不再称为论文要求的
   两路 INT4 O3。

因此本轮在“不改变 O3定义”的范围内结束优化，并如实报告目标未达到及其硬件原因。

## 9. 验收与复现

服务器仓库必须位于 `~/oyzl/ADAngel_oyzl`。本轮不需要重新采集或准备 trace。

```bash
cd ~/oyzl/ADAngel_oyzl
conda activate adangel-sm120
source scripts/activate_server_env.sh

ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . \
  --no-build-isolation --no-deps

python scripts/validate_o3.py
python -m pytest tests/integration/test_sm120_o3_o4.py -q --run-sm120

python scripts/validate_o3.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 \
  | tee reports/o3_4096_validation.json

EXTENSION_DIR=$(python -c \
  "import torch,pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit
python -m adangel doctor --require-native
```

服务器本轮诊断产物位于：

```text
reports/o3_target_half/iteration14/combined_24samples.json
reports/o3_target_half/iteration14/production_24samples.json
reports/o3_target_half/iteration14/production_4096_validation.json
reports/o3_target_half/iteration14/o3_combined_compute.ncu-rep
reports/o3_target_half/iteration14/production_audit/summary.txt
```

## 10. 官方资料

- [CUDA C++ Programming Guide 12.9.1：Compute Capability 12.0](https://docs.nvidia.com/cuda/archive/12.9.1/pdf/CUDA_C_Programming_Guide.pdf)：CC 12.0 Tensor Core原生整数矩阵类型列出 INT8，未列出 INT4。
- [PTX ISA：warp-level matrix instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)：定义兼容的 U4/S4 `mma.sync` PTX语义；PTX语义不等同于目标 GPU的独立 SASS opcode。
- [CUTLASS warp MMA definitions](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm80.hpp)：本项目使用的 sub-byte MMA wrapper与 PTX形状来源。

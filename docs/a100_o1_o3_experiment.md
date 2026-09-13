# A100 O0/O1/O3 对照实验：FP16 baseline 与原生 INT4

**新增 O0 结论**：同进程三路交错 GEMM median 为 O0 `0.603136 ms`、O1 `4.390912 ms`、
O3 `1.833984 ms`。当前 O3 快于 O1，但仍慢于 cuBLASLt FP16 O0。O0 转换与端到端测量、
数值验证和稳定性限制见“补充 FP16 O0 baseline”一节。

状态：SM80 编译、96 项正确性检查、内存检查、原生 INT4 指令审计和 24 样本测量已完成。
运行中出现外部 GPU 负载，192 条记录中 62 条至少一个阶段 CV≥3%；不能将这次测量
标为“全部稳定通过”。完整数据保留，下面给出带干扰说明的结果。
补充的 24 样本紧密交错 compute-only 测量已经完成：每个样本/后端合计 200 次，
48 组聚合计时均 CV<3%，O3/O1 配对吞吐比中位数 **2.4087×**。

## 范围与架构适配

O1 保持 E2M1 精确转 INT8 base，逐 K32 scale/FP32 累加；O3 保持 G128 MXFP4
转 Q4、INT8 激活拆成 low U4 与 high S4，分别进行 U4×S4、S4×S4 MMA，
合并 `low+16*high`，逐 G128 scale/FP32 累加。转换代码与 5090 共用。

A100 不具备 TMA。SM80 后端采用共同的 cp.async 双缓冲与 warp-cooperative
搬运/计算方式，O1 输出 tile=64×64、K stage=64，O3 输出 tile=64×64、K stage=128。
这属于 A100 上的成对移植实验，不能把与 5090 之间的差异全部归因于 INT4 指令。
O3 必须在同一个实际 kernel 的 SASS 中同时出现原生 U4×S4 和 S4×S4 IMMA。

### 本次 SM80 实现

| 项目 | O1 | O3 |
|---|---|---|
| 数值输入 | A8、精确 `2*E2M1` 的 W8 基值 | low U4 / high S4 激活、G128 Q4 权重 |
| CTA 输出 tile | 64×64 | 64×64 |
| 每个 pipeline stage 的 K | 64，包含两个 K32 | 128，包含两个 K64 MMA 子块 |
| CTA 线程 | 256（8 个 warp，4×2 warp 布局） | 相同 |
| 搬运与同步 | 两阶段 cp.async；所有 warp 协作 | 相同策略，额外搬运 high 激活 |
| INT32 partial | 寄存器内，K32 后清零并缩放 | 两路寄存器 partial，G128 后合并并缩放 |
| FP32 累加 | 每 K32 一次 | 每 G128 一次 |
| 最终输出 | 每个输出元素一次 global store | 相同 |

O1 对每个 K32 计算 `partial * (A_scale * W_scale / 2)` 并进行 FP32 FMA。
O3 对每个 G128 计算 `(low_partial + 16*high_partial) * (A_scale * W_scale)`
并进行 FP32 FMA。共享内存只用于输入双缓冲，不保存中间输出矩阵。
4-bit 输入通过 CuTe subbyte iterator 寻址，避免将 nibble 的逻辑步进误解释为字节步进。

这不是把 5090 production kernel 原封不动换一个编译参数：A100 不具备其 TMA 路径，
因此使用独立移植实现。SM80 本轮没有移植 5090 的全部 CTA/scale 共享/展开优化，
也没有做 A100-specific autotune；结果代表这里明确列出的实现，不代表 A100 极限性能。

### 已通过的验证

- 96 项语义对照：零值、完整位模式/饱和值与随机数据，四种模式。
- 覆盖 64×64×128、128×192×256、192×128×384、128×128×4096。
- 对逐组软件参考满足 `rtol=1e-3, atol=1e-3`，输出为 finite FP32。
- Compute Sanitizer memcheck：96 项检查通过，`ERROR SUMMARY: 0 errors`。
- 对各自实际 GEMM function 审计，而非匹配 probe：

| 审计项 | O1 | O3 |
|---|---|---|
| 实际 SASS | `IMMA.16832.S8.S8` | `IMMA.16864.U4.S4`、`IMMA.16864.S4.S4` |
| 静态 IMMA 指令位置数 | 8 | 8+8 |
| LDGSTS 指令位置数 | 24 | 36 |
| 寄存器/线程 | 80 | 80 |
| Shared/CTA | 24576 B | 24576 B |
| Stack / local memory | 0 / 0 | 0 / 0 |

指令位置数是反汇编中的**静态计数**，不是整个 4096³ GEMM 的动态执行次数。
O3 对应 function 中没有 INT8 IMMA；本次 A100 运行确实走了原生 U4/S4 路径。

## 服务器环境

全部项目、环境、缓存、临时文件均位于 `/home/zlouyang` 下。
系统已有 CUDA 12.8.93、驱动 570.124.06、A100 PCIe 40GB。
使用用户目录 Miniconda，Python 3.10、PyTorch 2.7.1 cu128、固定 CUTLASS 4.5.2。

```bash
export TMPDIR=/home/zlouyang/tmp
export PIP_CACHE_DIR=/home/zlouyang/.cache/pip
export PATH=/home/zlouyang/miniconda3/envs/adangel-a100/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
cd /home/zlouyang/ADAngel_oyzl
python -m pip install PyYAML==6.0.2 typing_extensions==4.12.2 numpy==2.1.3 ninja==1.11.1.3 setuptools==75.8.0 wheel==0.45.1 pytest==8.3.5
bash scripts/fetch_cutlass.sh
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps
```

编译生成 `adangel._sm80`；默认 `_sm120` 构建路径仍独立存在。

### 兼容原有 RTX 5090 服务器

这是新增 SM80 后端，不是把原项目改成仅支持 A100。`csrc/sm120/`、原有
`csrc/bindings.cpp` 和 5090 Python 调度入口继续保留；A100 使用自己的绑定和
`run_a100_experiment.py`。默认不设置 `ADANGEL_CUDA_TARGET` 时仍构建 SM120。

在 5090 的 `/home/zlouyang/oyzl/ADAngel_oyzl` 中，激活原 `adangel-sm120` 环境后执行：

```bash
source scripts/activate_server_env.sh
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm120 \
  python -m pip install -v -e . --no-build-isolation --no-deps
python -m adangel doctor --require-native
```

原有实验配置、prepared trace、审计脚本和正式运行命令无需改成 A100 版本。
构建目标隔离由 `tests/unit/test_cuda_build_targets.py` 验证；此测试只验证构建选择，
不能代替两台 GPU 各自的编译、正确性与指令审计。

5090 实机兼容性已验证：同步新增 SM80 源码后，未设置 `ADANGEL_CUDA_TARGET`
进行默认重新编译成功；`doctor --require-native` 通过，O0/O1/O2/O3/O4、O3 INT8×2
及构建选择相关测试合计 131 项通过。原有 SM120 CUDA 源码和绑定没有被替换。

## 验证、审计、真实数据运行

```bash
python scripts/run_a100_experiment.py --validate-only --output reports/a100_validation
python scripts/audit_sm80.py --output reports/a100_audit
python scripts/run_a100_experiment.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 --output runs/a100_o1_o3 \
  --warmup 50 --repeats 200 --inner 100
```

使用已有 24 个 prepared trace，逐文件校验 SHA-256；无需下载或加载大模型。
每个样本先用共用的 cuBLASLt O0 后端得到 FP16 输入、FP32 累加/输出参考。
O1/O3 交错测试四种模式，转换阶段摊销 100 次，GEMM 与端到端采用单次直接计时。
保存环境、编译 commit、输入 manifest、全部原始 timing、各阶段统计、FP64 reduction MSE。
CV≥3% 的记录保留并列出，不通过剔除数据制造稳定结果。

验收需同时具备：合成边界数据正确性、真实数据 finite FP32 输出与 MSE、
同入口 SASS 原生 INT4 证据、24 样本成对计时及对实现差异的说明。

## 24 个真实样本的结果

运行目录：`runs/a100_o1_o3_native_v1`。24 个样本 × O1/O3 × 4 个模式，共 192 条记录。
每项延迟先对单一样本的 200 次测量取 median，再对 24 个样本取 median / mean。
所以“Mean ms”是**24 个样本 median 的均值**，不是全部 4800 次耗时的算术平均。
未删除、裁剪或替换任何离群记录。以下数字反映本次 SM80 移植代码与当时运行条件。

### 转换开销（conversion-only）

| 后端 | W 转换 median ms | A 转换 median ms | 转换合计 median ms | 转换合计 mean ms |
|---|---:|---:|---:|---:|
| O1 | 0.064015 | 不需要 | 0.064015 | 0.064076 |
| O3 | 0.059479 | 0.041047 | 0.100590 | 0.100307 |

O1 的 W 转换为 E2M1→INT8 基值；O3 的 W 转换为 E2M1→Q4，A 转换为 INT8→两路
packed U4/S4。G128/G32 公共权重量化已在 prepared trace 中完成，不计入这里。
本脚本 conversion-only 的合计按每次独立摊销测量的 W/A 时间相加，再进行统计；
它不是一次 cold 调用的直接耗时。缓存状态、事件开销与不同统计量都可能使各表不能直接相加。

### GEMM-only（compute-only）

| 后端 | Median ms | Mean ms | 24 样本配对吞吐比（O3/O1） |
|---|---:|---:|---:|
| O1 | 4.457984 | 4.441003 | 参考 |
| O3 | 1.871872 | 1.997056 | 2.3821× |

每个样本的吞吐比定义为 `O1 GEMM median / O3 GEMM median`，表中取 24 个配对比的
中位数；不是先平均再相除。当前测量下 O3 延迟约为 O1 的 42%，不是“O3 只有 O1 一半吞吐”。
不过，这**不是原生 INT4 必然获得 2.38× 加速的硬件定律**，见下文边界说明。

### Cold total

| 后端 | Median ms | Mean ms |
|---|---:|---:|
| O1 | 4.529664 | 4.510101 |
| O3 | 1.982464 | 2.182592 |

直接计时当前配置所需的 W 转换、在线 A 转换和 GEMM；不包含 CPU 文件加载、显存申请、
原始 FP16 trace 公共准备或 cuBLASLt 算法搜索。

### Steady-state total

| 后端 | Median ms | Mean ms |
|---|---:|---:|
| O1 | 4.464128 | 4.449131 |
| O3 | 1.918976 | 1.911083 |

W 转换已缓存。O1 在线只运行 GEMM；O3 在线还需要拆分激活。总时间用单次直接 CUDA
Event 区间测量，不用独立转换时间与 GEMM 时间相加代替。

### 输出 MSE（相对本机 O0）

| 后端 | MSE median | MSE mean |
|---|---:|---:|
| O1 | 9.823388e-09 | 1.114204e-08 |
| O3 | 6.653010e-03 | 7.578847e-03 |

每个样本按 `mean((Y_variant.double()-Y_O0.double())**2)` 计算。O0 在同一 A100 上使用
FP16 输入、FP32 累加/输出的 cuBLASLt 路径。不是权重/激活本身的误差，也不是下游任务精度。
O1 的 E2M1→INT8 基值转换精确；很小的 MSE 来自参考 FP16 表示及累加/缩放路径差异。
O3 另外包含 G128 分组与 E2M1→Q4 舍入误差，原生 INT4 指令不会消除这些量化误差。

## 紧密交错的 compute-only 复核

为减少前一次运行中长时间块状测量受到的时间漂移影响，另跑同样的 24 个样本。
每个样本进行 10 轮，每轮每个后端预热 5 次、计时 20 次，O1/O3 顺序逐轮交替。
每个样本/后端仍累计 50 次预热、200 次测量，但分散到交错的 10 轮中。
输入转换、显存分配和 MSE 计算均在 GEMM 计时外，没有更换 kernel 或数学语义。

| 后端 | 24 样本 GEMM median ms | 单样本 200 次聚合 CV 范围 | CV≥3% 的样本数 |
|---|---:|---:|---:|
| O1 | 4.373504 | 0.134%–0.820% | 0/24 |
| O3 | 1.817088 | 0.060%–1.701% | 0/24 |

先对每轮的 O1/O3 median 延迟配对，得到 `t_O1/t_O3`，再取样本内 10 轮 median，
最后取 24 样本 median，结果为 **2.408650×**。24 个样本的该比值范围为
`2.397026–2.410704×`，与全量四模式测量的 `2.382112×` 相近。

没有删除任何计时值：480 条“样本/后端/轮次”记录中有 1 条短轮 CV≥3%，仍保留在聚合中。
聚合 CV 稳定不能保证完全没有外部负载；每个样本开始时仍记录了 GPU 进程与时钟快照。
这组数据支持**当前 A100 移植实现下 O3 吞吐约为 O1 的 2.4 倍**，而不支持将该比值
当作普适硬件峰值比。它也没有覆盖转换/cold/steady-state，不能替代前一节的四模式数据。

复现命令：

```bash
python scripts/measure_a100_paired_compute.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/a100_paired_compute_new \
  --rounds 10 --repeats 20 --warmup 5
```

## 如何理解性能差异

1. **本机 O3 确实使用原生 INT4。** 实际 SASS 的 U4/S4 指令已审计，而不是仅凭 PTX
   名称或 capabilities 字段判断。没有采用先扩展 INT8 再执行的 5090 lowering 路径。
2. **两路计算不意味着总耗时必须是 O1 的两倍。** O3 两路 MMA 的 K 为 64，O1 的 K 为 32；
   同时 O3 只需 32 个 G128 软件缩放，O1 需要 128 个 K32 软件缩放。数据搬运、scale、
   同步和指令调度也计入 GEMM-only，不能只数“两条 MMA”。
3. **这是两套明确配置的 SM80 实现之比，不是完整硬件上限之比。** O1/O3 共用输出
   tile、线程数和 cp.async 策略，但不等于二者都已充分调优。特别是 O1 本轮没有移植
   5090 production 的更大 CTA、CTA 内 W scale 共享等全部优化。
4. **不能用本表直接证明 A100 比 5090 更快或更慢。** 两边 GPU、CTA、pipeline、warp
   调度和后处理代码不同。要隔离 INT4 硬件贡献，还需要在同一 A100 上做额外的匹配后端
   消融；本轮并未完成该因果隔离。
5. **外部负载限制绝对延迟的解释。** 运行中观测到另一个用户的 GPU 计算进程，随后
   部分阶段有明显长尾，甚至个别样本的中位数也受影响。62/192 条记录未达到 CV<3%，
   因此不能声称全量结果稳定验收通过；中位数也不是消除调度干扰的保证。

本轮完成的是原生路径与数值正确性验证，以及带运行条件说明的性能测量。若需要发布严格的
独占 GPU 延迟，应在资源协调后重新运行同一命令；不需要改代码、重采数据或删除本次结果。

## 补充 FP16 O0 baseline：同进程三路对照

O0 从与 O1 相同的 prepared INT8 激活、G32 MXFP4 权重出发，分别反量化成 FP16，
然后执行一次 cuBLASLt FP16×FP16、FP32 累加/输出 GEMM。**不是直接拿原始 FP16 trace
做乘法**，因此没有改变之前 O0 参考的数值定义。
复用 `_sm80.benchmark_o0` 中已有的 O0 实现，本次没有修改或重新编译 CUDA 后端。

### O0/O1/O3 交错 GEMM-only

运行目录为 `runs/a100_o0_o1_o3_paired_v1`。同一进程中对同一份 24 样本进行三路轮换：
每样本 10 轮，每后端每轮预热 5 次、测量 20 次，按三种后端的 6 种排列循环运行。
各后端每样本合计 200 次测量，未剔除任何计时值。

| 后端 | GEMM median ms | 配对吞吐 / O0 | 24 样本聚合 CV 范围 | CV≥3% 样本数 |
|---|---:|---:|---:|---:|
| O0：cuBLASLt FP16 | 0.603136 | 1.000000× | 0.487%–1.843% | 0/24 |
| O1：INT8/K32 | 4.390912 | 0.137073× | 1.066%–3.653% | 5/24 |
| O3：两路原生 INT4/G128 | 1.833984 | 0.327290× | 2.533%–5.685% | 22/24 |

吞吐比先在同一轮计算 `t_O0/t_variant`，再取每样本 10 轮 median 和 24 样本 median。
因此不要求它恰好等于表中两个跨样本 median 延迟的比。
本次三路测量的 O3/O1 配对吞吐比为 `2.403491×`，与前面的独立两路对照相近。

O0 的 24 组聚合计时均稳定，但本轮 O1/O3 有上述 CV 异常，不能称三路均通过 CV 门槛。
三路同进程运行、不同 kernel 的功耗/缓存状态、未锁频与可能的外部负载均属于本次测量条件；
未单独隔离其影响，不能把每一个波动都认定为外部进程导致。

### 本轮输出 MSE

| 后端 | MSE median | MSE mean |
|---|---:|---:|
| O0 | 0 | 0 |
| O1 | 9.823388e-09 | 1.114204e-08 |
| O3 | 6.653010e-03 | 7.578847e-03 |

本轮各样本最后一轮的 O0 FP32 输出作为参考，FP64 reduction；O1/O3 的结果与前一轮一致。
O0 的自比较 MSE=0 不是它相对原始未量化 FP16 模型完全无误差的证明。

### 为什么 O0 仍然更快

这组结果支持：**A100 原生 INT4 使当前 O3 快于当前 O1，但 O3 仍明显慢于 O0。**
O0 采用优化成熟的 cuBLASLt GEMM，反量化已移出 GEMM-only；其主循环不需要软件逐 K32/
G128 重置 partial、拆分重构和应用 scale。当前 SM80 O1/O3 则是前文明确列出的移植 kernel，
没有完成与 cuBLASLt 同等级的调优。较低位宽的单条指令潜在吞吐，并不等于整个复合 kernel
必然更快。这里不能据此声称“INT4 硬件比 FP16 硬件慢”，也不能把结果归因于单一瓶颈。

本次 O0 实际选用 cuBLASLt algorithm `6`，workspace 为 `0`，split-K 为 `1`，
`numerical_impl_flags=66050`，`CUBLAS_COMPUTE_32F`，FP16 输入、FP32 输出。
后端在调用前根据 cuBLASLt numerical implementation flags 检查 HMMA/FP16/FP32 属性；
这里记录的是库算法属性，不是另外一次 O0 SASS 反汇编审计。

复现三路对照：

```bash
python scripts/measure_a100_paired_compute.py --include-o0 \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/a100_o0_o1_o3_paired_new \
  --rounds 10 --repeats 20 --warmup 5
```

本节证据：[三路汇总](evidence/a100_o0/paired_summary.json)、
[720 条逐轮原始记录及 O0 算法信息](evidence/a100_o0/paired_rounds.jsonl)、
[测量配置](evidence/a100_o0/paired_environment.json)。
本次三路测量脚本版本为 `2b488a7`；下面 O0 独立四模式脚本版本为 `3ed9dcf`，
具体完整 commit 和相同的二进制 SHA-256 见各自 environment.json。

### O0 独立四模式测量

运行目录为 `runs/a100_o0_baseline_v2`；每样本预热 50 次、测量 200 次，转换 inner=100。
仍为 24 个样本，下面每个 median/mean 先基于单样本的 median 再跨样本汇总。
这些数据与上面的三路交错测量属于**不同运行与调度口径**，不相互替换或跨表相加。

#### Conversion-only

| 阶段 | Median ms | Mean ms |
|---|---:|---:|
| MXFP4 权重→FP16 | 0.063130 | 0.062967 |
| INT8 激活→FP16 | 0.049167 | 0.048985 |
| W+A 转换合计 | 0.112845 | 0.112406 |

W/A 各自在一个 CUDA Event 区间中重复 100 次后摊销。O0 的转换合计是
**W+A 两个 kernel 一起重复 100 次的独立测量**，不是两个独立 median 之和。
它与前文 SM80 O1/O3 脚本“独立阶段时间相加”的 conversion-only 合计定义有区别；
端到端对比均应采用直接测量的 cold/steady-state，而不是这两个合计做强行比较。

#### Compute-only

| 阶段 | Median ms | Mean ms |
|---|---:|---:|
| FP16 GEMM | 0.761344 | 0.757888 |

#### Cold total

| 阶段 | Median ms | Mean ms |
|---|---:|---:|
| W 反量化 + A 反量化 + GEMM | 0.852480 | 0.842411 |

#### Steady-state total

| 阶段 | Median ms | Mean ms |
|---|---:|---:|
| 缓存 W_fp16，只执行 A 反量化 + GEMM | 0.800256 | 0.797355 |

**稳定性说明**：O0 独立四模式的 96 条记录中有 57 条至少一个阶段 CV≥3%。
分别为 conversion-only `0/24`、compute-only `18/24`、cold `20/24`、steady-state `19/24`。
数据原样保留，不能标记为全量 CV 验收通过。

该运行的样本起始快照从约 `1395 MHz / 50°C` 变化到 `1320 MHz / 70°C`；三路交错运行的
对应首尾快照均为 `1410 MHz`、温度约 `40→60°C`。两次运行这些快照未见其他计算进程。
时钟/温度、测量持续时间以及不同前序工作可能影响结果，但这些快照并未覆盖每个 kernel，
所以没有证据将 `0.761344 ms` 与 `0.603136 ms` 的差额全部归因于某一因素。
**横向比较 O0/O1/O3 时优先使用同次三路交错表；四模式表用于如实记录本次完整路径开销。**

### O0 正确性与复现

- 16 项合成检查通过：128×192×256 与正式 4096³，随机/全零输入、四种模式。
- 所有 24 个真实样本、四种模式均与独立参考对照通过。
- 反量化 A/W 与 Python FP16 参考逐元素一致；输出对 FP32 matmul 参考满足
  `rtol=1e-3, atol=1e-3`，参考 matmul 禁用 TF32。
- 真实数据最大绝对差为 `0.0010986328125`，合成检查为 `0.00019073486328125`。
  验收使用上述相对/绝对组合容差，不能误写成所有绝对误差均小于 0.001。
- 输出为 finite FP32；四模式共 96 次调用均选择上述 algorithm 6。
- 使用同一 prepared manifest，SHA-256：
  `05849422f6d8ad18e7c3468ff6af549d33ce86eeb9590af7388ab3452d26881c`。
- `_sm80` 二进制 SHA-256 与前轮完全一致；本次仅增加/扩展 Python 测量脚本。

```bash
python scripts/measure_a100_o0.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/a100_o0_baseline_new \
  --warmup 50 --repeats 200 --inner 100
```

证据：[96 条原始四模式记录](evidence/a100_o0/full_results.jsonl)、
[汇总及所有 CV 异常](evidence/a100_o0/full_summary.json)、
[16 项合成检查](evidence/a100_o0/validation.json)、[环境](evidence/a100_o0/environment.json)、
[24 样本输入 manifest](evidence/a100_o0/data_manifest.json)、
[四模式 GPU 快照](evidence/a100_o0/full_gpu_snapshots.jsonl)、
[三路交错 GPU 快照](evidence/a100_o0/paired_gpu_snapshots.jsonl)。

## 可核查证据与复现

前轮 O1/O3 原生 INT4 四模式测量的源码 commit 为 `e1ca46a542e3d21f66edc5912663d0f25f253ab9`，
`_sm80` 二进制 SHA-256 为
`8a3f6ffed76cb13efd45a80db7a2e28033bb450de56267383018c51453663f5e`。
后续只补充报告/测量脚本，不把不同 kernel 的结果混在一起。

- [全量 192 条原始计时、MSE 与 kernel 元数据](evidence/a100_o1_o3/full_results.jsonl)
- [四种模式汇总与所有 CV 异常记录](evidence/a100_o1_o3/full_summary.json)
- [环境及实验参数](evidence/a100_o1_o3/environment.json)
- [96 项正确性检查](evidence/a100_o1_o3/validation.json)
- [Compute Sanitizer 零错误记录](evidence/a100_o1_o3/memcheck.txt)
- [指令审计摘要](evidence/a100_o1_o3/audit.json)
- [O1 完整 function SASS](evidence/a100_o1_o3/o1.sass)、[O3 完整 function SASS](evidence/a100_o1_o3/o3.sass)
- [编译资源用量](evidence/a100_o1_o3/resources.txt)
- [5090 默认重编译后的 131 项测试](evidence/a100_o1_o3/rtx5090_tests.txt)、[doctor](evidence/a100_o1_o3/rtx5090_doctor.json)
- [交错测量汇总](evidence/a100_o1_o3/paired_summary.json)、[全部 480 轮记录](evidence/a100_o1_o3/paired_rounds.jsonl)、[交错测量配置](evidence/a100_o1_o3/paired_environment.json)

完整 PTX/SASS、构建/内存检查日志和 GPU 进程快照在本地 `reports/a100_native/` 及 A100
对应 `reports/a100*`、`runs/a100_*` 下保留。公开证据不包含其他用户进程的完整路径。

全部修改先在本地 `/root/ADAngel_oyzl` 完成并推送 GitHub。A100 使用 `fetch` +
`merge --ff-only` 同步；HTTPS 间歇失败时，用已推送分支生成的 Git bundle 作为传输载体，
经 `git bundle verify` 后仍执行 fetch/快进合并，没有直接编辑 A100 上的实现。
A100 的环境、源码、数据、构建和报告写入均限制在 `/home/zlouyang`；5090 操作限制在
`/home/zlouyang/oyzl/ADAngel_oyzl`。没有改驱动、系统 CUDA、时钟设置或其他用户进程。

参考：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，
`mma` 的 SM80 sub-byte 形状与 `cp.async` 异步数据搬运说明。

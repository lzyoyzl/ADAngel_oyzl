# O5/O6 A100：实现契约与当前状态

当前状态：正在实现；正式 O5/O6 尚未启用。不能用整数合成输入的
compute-only 验证代替格式量化、真实 trace 的 MSE 或端到端验收。

共用整数 core 已在 A100 编译并通过首轮数值、原生 INT4 指令、memcheck
和 racecheck 验证，见 [原始证据及范围说明](evidence/a100_split_grouped_v1/README.md)。
较大 tile 的合成 GEMM median 为 0.631808 ms、CV 为 3.935%，仅作初筛。
格式与 F 选择已确认。A100 现有数据是 prepared，不是原始 FP16；用户要求先
核对与旧实验的对齐方式，目前原始 trace 的提供/传输与二次量化取舍均未确认。

2026-09-29 格式→CUDA 转定点→整数 GEMM 的合成验证已通过，包括 51 项
转换逐位对照、36 项 GEMM、24 项四模式接口及 memcheck/racecheck。
见 [原始证据及范围说明](evidence/a100_mixed_formats_v3/README.md)。
这仍不是正式性能验收；不宣称达到快于 O0 的目标。

`5046507` 的 group-major scale 候选已编译并完成合成检查：102 项转换、
72 项新旧布局逐位对照、36 项 GEMM、24 项计时契约，以及 memcheck/racecheck。
4096³ 三轮交错四模式结果和 NCU 结构对照见
[候选验收记录](evidence/a100_mixed_group_major_v1/README.md)。
GEMM 有约 5% 的初筛收益，但转换变慢，且共享 GPU 下有 CV 失败记录；
自然布局仍为默认，不能据此宣布 24 样本正式验收完成。

### 与旧实验完全对齐：原始 FP16 直接量化入口

旧 `prepare_trace.py` 从同一份 raw FP16 分别生成 INT8 A、MXFP4-G32 W、
MXFP4-G128 W；G128 并不是从 G32 反量化再量化而来。要对齐该起点，O5/O6
必须从原始 FP16 分别生成已批准的源格式，不能把 O0 展开的 FP16 称作原始值。

`benchmark_a100_mixed_trace.py --raw-data ...` 已加入此路径。它保留现有
prepared 的 O0/O1/O3 输入及 O0 参考，O5/O6 则使用 raw activation/weight。
源格式仅在内存生成，不落盘，公共量化在计时外；后续源格式→定点、scale、
packing 与 GEMM 仍沿用下面的四模式计时。没有修改旧后端或 prepared 文件。

运行前要求：raw manifest SHA-256 与 prepared 的 `source_trace` 完全一致；
raw 深度验证通过；逐样本在 CPU 重放旧准备步骤，所有 A/W 编码、scale、Q4
均与已有 prepared 逐位一致。张量必须原本是 contiguous finite FP16，禁止
先强制类型转换来掩盖错误。输出 MSE 仍相对旧 O0，不换成 raw FP16 GEMM。

数据到位且来源确认后使用（**尚未在真实 raw 上执行**）：

```bash
python scripts/benchmark_a100_mixed_trace.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --raw-data data/raw/llama2_7b_prefill --validate-input-only

python scripts/benchmark_a100_mixed_trace.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --raw-data data/raw/llama2_7b_prefill \
  --output runs/mixed_trace_original_v1 \
  --samples 24 --rounds 3 --warmup 50 --repeats 200 --inner 100 \
  --scale-layouts row_major group_major
```

该路径记录 `input_policy=original_fp16_direct_source_quantization` 和原始
manifest、原始操作数 hash；输入误差相对 raw 计算。不能与二次量化记录混合
汇总。A100 目前没有该 raw 目录中的张量；未自动传输数据或擅自改选实验口径。

原始路径已在 A100 通过 33 项单测及含 4096³ 的 84 条合成四模式记录，
旧二次量化入口另有 56 条回归记录；见
[原始 FP16 入口合成验收](evidence/a100_mixed_trace_original_v1/README.md)。
仍不替代真实 raw 的完整 CLI 与 24 样本正式性能验收。

### 备选：prepared 二次量化入口（仍需明确确认）

`scripts/benchmark_a100_mixed_trace.py` 同时保留一个**显式二次量化入口**，
不是把 prepared 文件冒充 raw FP16。它需要 `--allow-secondary-quantization`
才能执行，当前仍等待用户确认该路径。默认不自动生成或保存任何源格式数据。
不使用该授权标志时，只可运行以下只读校验：

```bash
python scripts/benchmark_a100_mixed_trace.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --validate-input-only
```

校验包含 24 样本 manifest 契约、实际 `.pt` 文件集合、路径与每个文件的
SHA-256；此命令不加载模型、不量化、不编译，也不写数据。张量内容验证在
真正运行时由 `load_prepared`、原生参数检查及逐位/参考输出检查完成。

若用户确认二次量化，运行路径将固定为：

```text
已有 A_int8 + A_scale / W_mxfp4(K32) + W_scale
  → O0 原生反量化的 FP16 操作数（并与独立软件解码逐位核对）
  → 在内存中生成 O5/O6 源格式（计时外，不保存新 .pt）
  → 源格式转定点、scale、packing（计时内）
  → 双 INT4 GEMM、FP32 输出
```

这不能恢复原始 FP16 已损失的信息。O0/O1 使用原有 K32 prepared，O3 使用
原有 G128 prepared，不被重生成或改写；O5/O6 起点是 O0 实际操作数。
因此该数据口径的结论不能写成“原始 FP16 直接量化为 NVFP4/HiF4”的精度结果。

确认后才使用以下命令（没有在真实数据上执行）：

```bash
python scripts/benchmark_a100_mixed_trace.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/mixed_trace_secondary_v1 \
  --allow-secondary-quantization \
  --samples 24 --rounds 3 --warmup 50 --repeats 200 --inner 100 \
  --scale-layouts row_major group_major
```

默认只选择 row-major；这里显式选择两种布局用于配对验收。O0/O1/O3、
两种 O5 与两种 O6，共 7 个 case×24 样本×4 模式×3 轮=2016 条记录。
当前自然布局不是因这条命令而被替换。共享 GPU、未锁频，CV 超限只标记，
不过滤、不自动重跑到通过。Compute-only 对比 `gemm` 区间；旧 O1/O3 的
native total 包括原有额外 Event 标记，原始值保持不变。

输出文件：`config.json`、`environment.json`、`data_manifest.json`、
`source_provenance.jsonl`、`source_formats.jsonl`、`validation.jsonl`、
`results.jsonl`、`gpu_snapshots.jsonl`、`summary.json`。
source 文件只记录编码张量的 shape/dtype/hash 与误差指标，不保存编码张量。
每个样本先对独立有序 G128 定点参考验算，再运行四模式；模式间输出要求
逐位一致。跨布局也要求逐位一致，不仅要求 MSE 接近。

W/A 分别报告 source vs bridge、fixed vs source、fixed vs bridge 的 MSE；
这些误差不可相加。主输出 MSE 仍比较 O0。汇总先折叠同一样本的重复轮次，
再统计 24 样本的 median/mean；速度比按同样本同轮配对。Bootstrap 重采样
单位为样本，不把 200 次 Event 或 3 轮视为独立样本；同一 trace 中的样本
存在相关性，置信区间只作描述，不代表独立模型/语料的泛化结论。

`scripts/validate_a100_mixed_trace_runner.py` 仅用合成 prepared 输入验证以上
运行引擎。它不读取真实数据目录，也不能取代正式 24 样本结果。
该入口已在 A100 通过 256³/512³/4096³ 合成检查，共 84 条四模式记录；
另有 30 项相关单测通过，真实 prepared 的 24 文件 hash 只读校验通过。
详见 [入口验收证据](evidence/a100_mixed_trace_runner_v2/README.md)。

2026-09-28 已完成精度项目/O3 转换语义、A100 数据目录和外部格式资料的核对，
见 [格式与数据核对记录](o5_o6_format_review.md)。仍需澄清是否只不另存中间
格式，还是完全跳过源格式量化。前者属于 prepared 二次量化实验，后者只能
验证整数后端。当前推进编码/转定点正确性；不提前发布正式 O5/O6 结论。

## 实验计算路径（输入来源按最新授权待落定）

| 项目 | O5 | O6 |
|---|---|---|
| 权重目标格式 | NVFP4，G128 实验变体 | HiF4，G128 实验变体 |
| 激活目标格式 | MXFP8 E4M3，G128 实验变体 | NV-style FP6 E2M3，G128 实验变体 |
| 转定点 | 权重 Q4、激活 Q8 | 权重 Q4、激活 Q6，符号扩展至 INT8 |
| 整数计算 | low U4×S4 + 16×high S4×S4 | 同左 |
| 累加/输出 | G128 有序 FP32 FMA / FP32 | 同左 |
| 目标 | A100 compute-only 快于同进程 O0 | 同左 |

INT6 的 padding 必须为符号扩展。例如 -1 的六位补码 111111 变为
INT8 的 11111111，不能补两个零变成 +63。两路 INT4 指两种乘积路径，
不是整次 GEMM 只执行两条机器指令。

## 已确认的转定点契约

| 路径 | 源格式 | F | 整数的有效 scale |
|---|---|---:|---|
| O5 W | E2M1 + E4M3/FP32 两级 scale | 0 | 原两级 scale 乘积 |
| O5 A | E4M3 + UE8M0 | -2 | UE8M0 数值 × 4 |
| O6 W | S1P2 合并 8/4 元素微指数 + E6M2 | 0 | E6M2 基础 scale |
| O6 A | E2M3 + E4M3/FP32 两级 scale | 2 | 原两级 scale 乘积 ÷ 4 |

q=clamp(RNE(v×2^F), −L, L)，L=2^(Q−1)−1。F 固定为格式级参数，不按
当前数据重新拟合。初始 HiF4 量化保留作者 BF16 中间舍入和幅值 half-up，
与后续转定点 RNE 分开。所有 G128 改造及 NV-style FP6 都标注实验变体。
Python 参考与标量测试在 `mixed_formats.py` / `test_mixed_formats.py`；
它们不是计时后端，不会自动启用 O5/O6 production。

原生转换入口 `_sm80._convert_mixed_source(source)` 位于
`csrc/sm80/mixed_conversion.cuh`：融合 decode、RNE、补码 packing 和 scale
补偿。Q4 输出 `[R,K/2]`，Q8/Q6 输出 `[2R,K/2]` 的 low-U4/high-S4 分面。
INT6 先转为有符号整数，再按 INT8 补码分面。所有合法源编码的局部数值范围
均不会触发对称整数饱和；非法/NaN 编码和有效 scale 溢出在 launch 前拒绝。
该入口用于验证，分配和检查不是转换性能。
`_sm80._benchmark_mixed(variant, mode, weight_source, activation_source, warmup,
repeats, conversion_inner_repeats, tile)` 在全部检查/分配后复用缓冲区执行
转换与 GEMM。四模式计时契约已接入，仍不等于正式数据性能验收。

| 模式 | 已计时的转换 | GEMM | total |
|---|---|---|---|
| conversion_only | W 和 A，各批量摊销 | 区间外生成正确性输出 | 每次 W/A 摊销样本相加，再统计 median/mean；对齐 A100 O3 |
| compute_only | 两侧提前转换 | 直接计时 | 与 GEMM 同一区间 |
| cold | W、A 单列为批量摊销结果 | 直接流程中的 GEMM 区间 | 单次 W+A+GEMM |
| steady_state | 缓存 W；A 单列为批量摊销结果 | 直接流程中的 GEMM 区间 | 单次 A+GEMM |

上述为 GPU CUDA Event 延迟，不包含 Python 格式校验、公共初始量化、
文件 I/O 或内存分配。转换微基准和端到端总时间不应强行相加对齐。

### 与前面实验对齐的计时规则（v2）

固定 warmup=50、repeats=200、conversion_inner_repeats=100、单 CUDA stream。
转换样本为同一 Event 区间连续执行 100 次后除以 100；GEMM 和 cold/steady
路径每个样本只执行一次。Event、缓冲区均在预热前创建；先测直接路径，再测
独立转换，避免长批量转换改变直接路径的初始缓存/频率状态。
这里的 cold 表示“未缓存静态权重转换”，不是清空 L2、冷启动进程或首次编译。

源格式公共准备（量化为 NVFP4/MXFP8/HiF4/FP6）、数据传输、分配、校验和 MSE
计算均在四种计时之外，与前面实验排除公共初始量化的原则一致。**不能排除**
源格式→定点、有效 scale 生成、激活拆分/packing 的在线开销：O5 的 Q8 转换、
O6 的 Q6 转换与符号扩展都属于 activation_conversion；W→Q4 和有效 W scale
生成属于 weight_conversion。steady-state 仅缓存 W 的这些结果。

历史实现有一项差别：O0/O2 的 conversion total 是联合批量 Event，A100 O3
和 SM120 O3/O4 是独立转换样本之和。为让**新 A100 O0/O3/O5/O6 对照表**可比，
`benchmark_a100_mixed.py` 统一用 `total[i]=W[i]+A[i]`（FP32 求和），然后对
这些 total 样本计算 median/mean；不是 `median(W)+median(A)`，也不代表实际
串联延迟。O0 原生联合 Event 结果另存 `native_total_timings_ms`，旧后端和旧
结果文件不改写。cold/steady 的 total 从不重构，一律保留单次直接 Event。

每条新记录携带 `timing_contract_version=2`、`total_timing`、
`native_total_timing`、`native_total_timings_ms` 和阶段 inner repeats。
版本 1 的 O5/O6 conversion total 不能直接混入版本 2 对照表；需重测。
MSE 仍为 FP32 输出相对同源 O0 的差值，用 FP64 reduction，完全排除在计时外。
保留原始 200 次数据、median/mean/P5/P95/IQR/CV；共享 GPU 下 CV≥3% 标注为
诊断异常，不删离群、不为凑阈值单方面重跑。NCU 耗时不混入 Event 性能。

计时契约 v2 已在 A100 重新编译并通过 24 项四模式接口检查、36 项 GEMM
正确性及一次 4096³ 四模式联调；原始记录见
[计时对齐验收](evidence/a100_mixed_timing_v2/README.md)。这不替代真实 24 样本
性能验收；该次合成联调有 CV≥3% 的记录，已保留并标注。

## 共用整数计算契约

已打包整数输入：

- A_split: uint8[2M,K/2]；前 M 行 low U4，后 M 行 high S4。
- W_q4: uint8[N,K/2]；每字节两个补码 S4。
- A_scale: FP32[M,K/128]。
- W_scale: FP32[N,K/128]。

每组执行：

    partial = sum(A_low * W_q4) + 16 * sum(A_high * W_q4)
    scale = round_fp32(A_scale[m,g] * W_scale[n,g])
    Y = fma_fp32(float(partial), scale, Y)

group 按 K 从小到大执行。scale 是转定点后整数单位对应的实数尺度，
包含必要的 2^-F 因子。接口只接受在 G128 内可用单一尺度表达的整数；
若格式含更细粒度尺度，前端须先正确对齐，不能直接套用。

相比原 O3 的每行激活 scale，O5/O6 的两侧 scale 都随 group 变化，
因此不能把激活 scale 整体移到 K 循环外，也不能默认它们是 2 的幂。

## 计算部分复用的优化

复用现有 O3 的 CuTe U4/S4 MMA、寄存器 partial、窄 N 范围的权重
fragment 复用、shared-memory swizzle、cp.async 双缓冲及最终向量写回。
新增每 CTA 每 G128 一次加载的 A/W FP32 scale，供消费者共享。
K256 pipeline 内仍分开处理两个 G128，不能跨 scale 相加后统一缩放。
A100 沿用 cp.async；不假称存在 TMA。

初始候选为 64×64×128、64×128×256。旧 O0–O4 的默认入口和选型不变。
此阶段不使用 magic bias 或指数位 scale 优化；它们不是格式正确性的前提。

新增内部候选 `scale_layout="group_major"`：有效 FP32 scale 在物理上按
`[G,rows]` 存放，Python 返回逻辑 `[rows,G]`、stride `[1,rows]` 的视图。
转换 kernel 直接写目标地址，不增加额外 buffer 或重排 kernel，相关寻址和
store 开销计入转换时间。GEMM 每个 warp 连续读取同组的行/列 scale。
该候选来自 NCU 对 scale 非合并加载的定位；默认仍为 `row_major`，收益和
数值/内存安全待 A/B 验收，不能提前视为已采用的 production 优化。

合成对照入口（需重新编译；输出目录必须不存在）：

```bash
python scripts/benchmark_a100_mixed.py --synthetic \
  --output runs/mixed_layout_ab --tiles 64x128x256 \
  --scale-layouts row_major group_major
```

NCU 专用 `--profile-case o5/64x128x256` 或
`o5/64x128x256/group_major` 只运行该目标，不运行参考 GEMM；须同时指定
`--modes compute_only --rounds 1 --repeats 1`。warmup50 时 O5/O6 的匹配 kernel
skip=51（含区间外初始化 GEMM），O3 skip=50。NCU 结果只用于瓶颈分析。

## 编译、验证与审计

在 A100 仓库和已配置环境中：

    ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 python -m pip install -v -e . --no-build-isolation --no-deps
    python scripts/validate_a100_split_grouped.py --output runs/split_grouped_validation
    python scripts/validate_a100_split_grouped.py --large --output runs/split_grouped_4096
    python scripts/audit_a100_o1.py --variant split_grouped --allow-spills --output reports/audit_split_grouped
    python -m unittest discover -s tests/unit -p test_mixed_formats.py -v
    python scripts/validate_a100_mixed_formats.py --output runs/mixed_formats_validation

输出须写入新目录。CUDA Event 区间只测已准备输入的 GEMM；量化、packing、
校验、分配均在区间外。合成数据结果明确标为 prepared_integer_core_only。
脚本覆盖独立 FP64 参考、逐组非二次幂 scale、INT6 负数、饱和值、
非默认 stream、非法输入和旧 O3 语义的逐位回归。
审计要求同一函数具有 cp.async/LDGSTS 以及原生 U4×S4、S4×S4 IMMA，
不得用 S8 或 probe 代替；spill 如实记录，不因允许 spill 放宽 ISA 条件。

格式验证脚本不读取真实 trace，使用确定性的合成 FP16 输入。其
`mse_vs_fixed_reference` 检查计算实现；`mse_vs_synthetic_o0` 展示源格式和
转定点引入的差异，不是正式 24 样本精度结果。合成测试无需确认正式数据
的存储方式，因此可以在数据口径澄清前独立完成。

后续正式验收仍需完整格式编解码、已确定来源的 24 个样本、MSE vs 同源 O0、
转换开销、四种计时模式、同进程 O0 配对性能和内存安全测试。
还需形成理论 peak/bottleneck、分层工作流和通用混合精度 insight 三份复盘。

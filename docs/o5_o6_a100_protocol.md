# O5/O6 A100：实现契约与当前状态

当前状态：正在实现；正式 O5/O6 尚未启用。不能用整数合成输入的
compute-only 验证代替格式量化、真实 trace 的 MSE 或端到端验收。

共用整数 core 已在 A100 编译并通过首轮数值、原生 INT4 指令、memcheck
和 racecheck 验证，见 [原始证据及范围说明](evidence/a100_split_grouped_v1/README.md)。
较大 tile 的合成 GEMM median 为 0.631808 ms、CV 为 3.935%，仅作初筛。
格式与 F 选择已确认；用户指定改用 A100 现有 prepared 数据，不再要求 raw 传输。

2026-09-29 格式→CUDA 转定点→整数 GEMM 的合成验证已通过，包括 51 项
转换逐位对照、36 项 GEMM、24 项四模式接口及 memcheck/racecheck。
见 [原始证据及范围说明](evidence/a100_mixed_formats_v3/README.md)。
这仍不是正式性能验收；不宣称达到快于 O0 的目标。

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
| conversion_only | W 和 A，各批量摊销 | 区间外生成正确性输出 | W+A 联合批量摊销，不是相加两个中位数 |
| compute_only | 两侧提前转换 | 直接计时 | 与 GEMM 同一区间 |
| cold | W、A 单列为批量摊销结果 | 直接流程中的 GEMM 区间 | 单次 W+A+GEMM |
| steady_state | 缓存 W；A 单列为批量摊销结果 | 直接流程中的 GEMM 区间 | 单次 A+GEMM |

上述为 GPU CUDA Event 延迟，不包含 Python 格式校验、公共初始量化、
文件 I/O 或内存分配。转换微基准和端到端总时间不应强行相加对齐。

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

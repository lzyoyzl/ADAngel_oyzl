# O5/O6 A100：实现契约与当前状态

当前状态：正在实现；正式 O5/O6 尚未启用。不能用整数合成输入的
compute-only 验证代替格式量化、真实 trace 的 MSE 或端到端验收。

共用整数 core 已在 A100 编译并通过首轮数值、原生 INT4 指令、memcheck
和 racecheck 验证，见 [原始证据及范围说明](evidence/a100_split_grouped_v1/README.md)。
较大 tile 的合成 GEMM median 为 0.631808 ms、CV 为 3.935%，仅作初筛。
格式契约仍待确认；原始 FP16 trace 位于 5090，复制到 A100 待用户授权。

## 已确认的实验目标

| 项目 | O5 | O6 |
|---|---|---|
| 权重来源 | FP16 → NVFP4，G128 实验变体 | FP16 → HiF4，G128 实验变体 |
| 激活来源 | FP16 → MXFP8，G128 实验变体 | FP16 → NVFP6，G128 实验变体 |
| 转定点 | 权重 Q4、激活 Q8 | 权重 Q4、激活 Q6，符号扩展至 INT8 |
| 整数计算 | low U4×S4 + 16×high S4×S4 | 同左 |
| 累加/输出 | G128 有序 FP32 FMA / FP32 | 同左 |
| 目标 | A100 compute-only 快于同进程 O0 | 同左 |

INT6 的 padding 必须为符号扩展。例如 -1 的六位补码 111111 变为
INT8 的 11111111，不能补两个零变成 +63。两路 INT4 指两种乘积路径，
不是整次 GEMM 只执行两条机器指令。

## 待用户确认，禁止默认为已确定

- MXFP8 元素采用 E4M3 还是 E5M2、缩放格式和量化舍入规则。
- NVFP4 的 G128 改造是否保留局部 E4M3 + 全局 FP32 两级 scale。
- NVFP6 采用 E2M3/E3M2 中哪一种，或其他指定编码及 scale。
- HiF4 的具体定义及 G128 改造；若存在内部 micro-exponent，必须明确
  如何对齐到统一 Q4。不能丢掉内部指数，或直接当作 E2M1。
- 转定点 q=clamp(RNE(v×2^F)) 的 F、clamp 边界及 scale/2^F 归属。
  不同动态范围不能无条件套用 E2M1 的 F=Q-4。

这些选择直接影响 MSE，因此此时不输出正式 O5/O6 精度或性能结论。

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

输出须写入新目录。CUDA Event 区间只测已准备输入的 GEMM；量化、packing、
校验、分配均在区间外。合成数据结果明确标为 prepared_integer_core_only。
脚本覆盖独立 FP64 参考、逐组非二次幂 scale、INT6 负数、饱和值、
非默认 stream、非法输入和旧 O3 语义的逐位回归。
审计要求同一函数具有 cp.async/LDGSTS 以及原生 U4×S4、S4×S4 IMMA，
不得用 S8 或 probe 代替；spill 如实记录，不因允许 spill 放宽 ISA 条件。

后续正式验收仍需完整格式编解码、24 个 FP16 trace、MSE vs O0、
转换开销、四种计时模式、同进程 O0 配对性能和内存安全测试。
还需形成理论 peak/bottleneck、分层工作流和通用混合精度 insight 三份复盘。

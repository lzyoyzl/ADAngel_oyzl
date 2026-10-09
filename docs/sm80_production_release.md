# A100 O3 / O7 / O8 最佳方案正式接入

## 应用范围

本次将已验收的独立候选接入 `adangel._sm80` 正式默认入口，不修改 RTX 5090 的 `_sm120` 后端。
O0/O1、O5/O6、O9/O10 的实现不变；量化值、G128 分组及 MSE 参考不变。

| 后端 | 正式 GEMM | 正式转换 | MSE 参考 |
|---|---|---|---|
| O3 | v89：全 K 安全整数累加、八链调度、M8 分组 CTA | 向量化 Q4/高低位转换，权重 factor/guard 准备 | O0 |
| O7 | v99：全 K 安全整数累加、八链调度、streaming 输出 | v139：packed NVFP4 权重＋MXFP8 warp lookup 激活 | O5 |
| O8 | 同一 v99 | v138：packed HiF4 权重＋packed FP6 激活 | O6 |

两路原生 `U4×S4` / `S4×S4` Tensor Core 保留。安全检查不通过的 tile 使用逐 G128 FP32 缩放路径，
不是溢出后继续计算。无效源编码直接报错。允许合理的 FP32 求和顺序变化，但必须重新核对 MSE。

全 K 快路径支持 K=4096、M 为64的倍数、N为128的倍数；O3 的该路径要求正常 UE8M0 code 1–254。
其他原有受支持形状仍采用原实现。`implementation="legacy"` 可显式复现切换前的正式调度。
旧候选的 `roof_tune` 显式选择不变，不会被新默认覆盖。

## 正式入口

```python
from adangel import _sm80 as native

# 不再需要实验目录中的临时 .so 或 cubin。
o3 = native.benchmark("o3", mode, A_int8, A_scale, W_mxfp4_g128, W_scale_g128,
                      warmup=1000, repeats=200, inner=100)
o7 = native._benchmark_mixed("o7", mode, weight_source, activation_source,
                            warmup=1000, repeats=200, conversion_inner_repeats=100)
# O8 使用相同接口，将 variant 改为 "o8"。

# 回归对照：在以上调用末尾加 implementation="legacy"。
```

`kernel.production_default=true` 是新默认标识；正式 symbol 为
`adangel_sm80_o3_fullk_grouped` / `adangel_sm80_o78_fullk_streaming`。
核函数资源、fallback tile 数及转换实现名称随结果保存。

## 计时口径

| 模式 | 计入内容 |
|---|---|
| Conversion-only | 权重转换/metadata＋激活转换/guard；每个 Event 区间重复100次，除以100 |
| Compute-only | 两侧数据及 metadata 已准备，仅 GEMM |
| Cold | 单次权重转换＋激活转换＋GEMM，直接 Event 计时 |
| Steady-state | 缓存权重及其 metadata；单次在线激活处理＋GEMM |

FP16→源量化格式的公共准备不计入上述开销。factor、norm、anchor、guard 等在线工作全部计入转换。
Diagnostic 数据导出在计时后进行；不能将独立计时的 Conversion 与 GEMM 简单相加代替端到端结果。

## 构建与验收

在 A100 的 `/home/zlouyang/ADAngel_oyzl`，使用现有 `adangel-a100` 环境：

```bash
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python setup.py build_ext --inplace --build-temp build/sm80-production

python scripts/audit_sm80_production.py --output reports/sm80_production_audit
python scripts/validate_sm80_production.py --validate-only \
  --output runs/sm80_production_validation

compute-sanitizer --tool memcheck --error-exitcode=99 \
  python scripts/validate_sm80_production.py --validate-only \
  --output runs/sm80_production_memcheck

python scripts/validate_sm80_production.py \
  --output runs/sm80_production_four24 \
  --rounds 3 --warmup 1000 --repeats 200 --inner 100
```

验证脚本中的 frozen cubin 仅用于与之前最佳版本做逐位对照，不是正式执行依赖。
该发布验收需保留 A100 既有的 v89/v99 codegen 目录；普通正式入口不需要这些目录。
小 M/N 测试只验证正确性、安全边界与非默认 stream，不用作性能筛选。

24 个真实样本逐样本交错测试新旧版本；每种模式3轮，每轮200次。记录原始计时、MSE、
GPU 状态和新扩展 SHA-256。保留所有 CV 超阈值的记录，不删除离群值。

## 本次验收结果

待 A100 构建、指令审计、安全检查及24样本正式测试完成后填写；此前候选数据不冒充本次正式结果。

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
import torch  # 先加载 PyTorch 的动态库
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
混合格式接口的旧 `tile="64x128x256"` 默认参数作为兼容选择器保留；当前正式核函数的实际
CTA 为 `64×128×128`，以返回的 `kernel.cta_tile` 为准。`scale_layout` 不再改变该快路径的物理布局。
旧 trace/synthetic 脚本仍可运行；INT4 与 Binary 的跨实现对照使用数值容差，重复同一实现仍检查逐位一致。

## 计时口径

| 模式 | 计入内容 |
|---|---|
| Conversion-only | 权重转换/metadata＋激活转换/guard；每个 Event 区间重复100次，除以100 |
| Compute-only | 两侧数据及 metadata 已准备，仅 GEMM |
| Cold | 单次权重转换＋激活转换＋GEMM，直接 Event 计时 |
| Steady-state | 缓存权重及其 metadata；单次在线激活处理＋GEMM |

FP16→源量化格式的公共准备不计入上述开销。factor、norm、anchor、guard 等在线工作全部计入转换。
Diagnostic 数据导出在计时后进行；不能将独立计时的 Conversion 与 GEMM 简单相加代替端到端结果。
Cold 表示本次不能复用转换后的权重，不是进程冷启动；显存分配、输入合法性扫描及文件读取不在计时内。

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

正式扩展已完成构建。以下检查通过：

- 正式 O3/O7/O8 symbol 同时包含两路原生 INT4 和异步搬运；GEMM SASS 与已选候选完全一致。
- 48 项新默认正确性检查（四模式、非默认 stream、正常与回退路径），以及3类非法编码拒绝。
- Compute Sanitizer 定向 memcheck：0 errors；范围为小 M/N、K=4096，不冒称全矩阵 sanitizer。
- 原 O1/O3 接口96项检查通过；原混合 trace 入口包含4096³的156条回归记录通过。

24样本、四模式、三轮新旧配对测试已完成，共 **1728条记录**。所有样本、所有模式的新默认输出均与
已选最佳候选逐位一致；与旧默认的逐样本 MSE 差异通过 `rtol=1e-5, atol=1e-12` 回归检查。

以下“旧默认”指本次切换前的正式实现，不是v142中已很接近最佳方案的v78内部候选。
延迟先取每样本3轮中位数，再汇总24样本；吞吐提升来自同样本、同轮配对比值，不直接用两个总体中位数相除。

### Conversion-only total

| 后端 | 旧默认 ms | 新默认 ms | 配对吞吐提升 |
|---|---:|---:|---:|
| O3 | 0.099817 | 0.047698 | +109.01% |
| O7 | 0.126011 | 0.062482 | +101.68% |
| O8 | 0.106778 | 0.058967 | +81.12% |

### Compute-only GEMM

| 后端 | 旧默认 ms | 新默认 ms | 配对吞吐提升 |
|---|---:|---:|---:|
| O3 | 0.570368 | 0.446464 | +27.82% |
| O7 | 0.635904 | 0.479232 | +32.69% |
| O8 | 0.643072 | 0.480256 | +33.54% |

### Cold total

| 后端 | 旧默认 ms | 新默认 ms | 配对吞吐提升 |
|---|---:|---:|---:|
| O3 | 0.684032 | 0.504832 | +35.46% |
| O7 | 0.777728 | 0.551936 | +40.47% |
| O8 | 0.766976 | 0.552704 | +38.38% |

### Steady-state total

| 后端 | 旧默认 ms | 新默认 ms | 配对吞吐提升 |
|---|---:|---:|---:|
| O3 | 0.624640 | 0.477184 | +30.98% |
| O7 | 0.703488 | 0.528896 | +33.49% |
| O8 | 0.704512 | 0.528384 | +34.33% |

### 新默认 MSE

| 后端 | 输出参考 | Median | Mean |
|---|---|---:|---:|
| O3 | O0 | 0.006653010287410 | 0.007578847013303 |
| O7 | O5 | 0.005536172273439 | 0.005053635851003 |
| O8 | O6 | 0.004411084910986 | 0.004381379299074 |

与已选最佳候选的 MSE 完全一致；与旧默认存在允许范围内的极小舍入差异，不能称新旧默认输出逐位相同。
唯一真实数据回退为 O8 `layer_24_o_proj` 的12/2048个CTA，与已选候选一致。

### 稳定性与资源边界

每侧、每后端、每模式均为72条记录。主指标CV≥3%的数量如下（旧/新）；所有原值保留。

| 后端 | Conversion total | GEMM | Cold total | Steady total |
|---|---:|---:|---:|---:|
| O3 | 0/0 | 3/0 | 2/0 | 0/3 |
| O7 | 0/0 | 2/0 | 20/0 | 3/0 |
| O8 | 0/0 | 3/1 | 10/0 | 1/2 |

**Cold中的独立W转换分项仍有较多CV≥3%：新O3/O7/O8分别为66/60/56条（各72条）。**
独立Conversion-only的各分项均稳定；转换收益以上面的Conversion-only为依据，Cold收益以直接total为依据。
本次不宣称所有分项严格满足CV<3%，也不将没有逐Event证据的波动确定归因于某一外部负载。
四模式配对加速比的bootstrap 95%区间下界均大于1；这仅说明本轮24样本的结果，不代表任意模型/形状。

三个正式核函数均为128线程、168寄存器/线程，资源允许3 CTA/SM。
O3 shared memory为50688 bytes、local size为16 bytes/线程，SASS有4条LDL和4条STL；这是已验收候选原有的少量spill。
O7/O8 shared memory为34304 bytes、local size为0，没有LDL/STL。没有以更换INT8指令来取得上述结果。

O0/O1、O5/O6、O9/O10未改实现源码。重编译时检查的63个旧模板中57个SASS逐位不变，6个Binary模板的机器码有变化；
相关明细保留，不能声称整个旧扩展二进制逐位未变。包含Binary的原入口回归已通过，本次未重新发表这些未优化后端的性能。

## 证据与复现状态

- A100：`NVIDIA A100-PCIE-40GB`；PyTorch `2.7.1+cu128`；CUDA `12.8`。
- 原生构建及性能测试提交：`fba110acf502c2eb303af340e463aa3c59d3b512`。
- 新扩展SHA-256：`31191693afe8a12779ad504ea30acdc69d42407a88b2bcd629b39723eccf0819`。
- [原始计时、汇总和环境](evidence/sm80_production_release_20261009/runs/sm80_production_four24/)
- [正式指令审计](evidence/sm80_production_release_20261009/reports/sm80_production_release/audit_r3/audit.json)
- [内存检查日志](evidence/sm80_production_release_20261009/reports/sm80_production_release/memcheck.log)

完整PTX/SASS仍保留于A100的 `reports/sm80_production_release/audit_r3/`；本地证据保存正式GEMM的SASS、资源、检查日志及原始计时。
测试期间一次SSH观察连接断开，但原测试进程继续执行，未重启或拼接重跑数据。

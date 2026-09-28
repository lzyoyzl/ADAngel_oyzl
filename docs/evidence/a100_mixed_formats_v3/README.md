# A100 O5/O6 格式、转换与四模式接口：合成验证证据

日期：2026-09-29。**这是合成正确性/接口验收，不是 24 样本正式实验完成报告。**
没有读取、重写或重新生成用户的 prepared trace，也没有传输原始 FP16 trace。

CUDA 构建源码：`4e100e6`；验证脚本版本：
`3dbfa52d26f3bfcc0daa9a79cb7f85e15eaf7717`。
二者之间只有脚本与文档变化。二进制 SHA-256：
`8efc4abc7ab0510ae9ee50e207a77c171c964a4fdff975931c5389219626805b`。
ISA 审计与验证记录指向同一二进制。

## 已验证的内容

| 检查 | 结果 | 边界 |
|---|---|---|
| 编码/格式参考单测 | 15 项通过，无跳过 | CPU/CUDA、全部有限编码/中点、HiF4 微指数与 Q6 符号扩展 |
| 原生 CUDA 转换 | 51 项通过 | 打包字节与补偿 scale 均与参考逐位一致，包含尾 CTA |
| 双 INT4 GEMM | 36 项通过 | 对独立固定整数语义参考最大绝对误差和 MSE 均为 0 |
| 四模式接口 | 24 项通过 | 阶段存在性、采样数、批量次数、缓存语义、结果正确性 |
| 非法输入拒绝 | 17 项通过 | CPU、NaN scale、shape/dtype、有效 scale 溢出 |
| 指令审计 | 原生 INT4/cp.async 通过 | 同函数 U4×S4、S4×S4、LDGSTS；没有 S8 替代 |
| Compute Sanitizer memcheck | 0 errors | 完整合成验证脚本 |
| Compute Sanitizer racecheck | 0 errors、0 warnings、0 hazards | 同上 |

GEMM 形状为 64×128×256、128×128×512、512×512×4096；每种形状覆盖随机、
全零和正负交替，并验证两个 CTA 候选。小 M/N 配 K4096 时，严格 O0
cuBLASLt 选择器找不到合规 HMMA 算法，因此该项使用 512×512×4096，
没有放宽 O0 Tensor Core 要求或退回软件参考。

`mse_vs_fixed_reference=0` 仅证明整数计算正确。记录中的
`mse_vs_synthetic_o0` 来自同一组合成 FP16 构造的 O0，通常非零；
它既不是 24 个真实样本的 MSE，也不代表实际模型精度。

## 资源和性能解释

| CTA | REG | STACK | 审计说明 |
|---|---:|---:|---|
| 64×64×128 | 90 | 0 | 无 local load/store，严格检查通过 |
| 64×128×256 | 128 | 8 | 仍有小量 spill；按既有用户许可作为 warning 保留 |

审计 `passed=true`，但 `strict_passed=false`，不能称为零 spill。
resource 输出的 SHARED:0 是静态 shared；GEMM 的动态 shared 数量见
validation.json 各 kernel 的 `shared_memory_bytes`，不是没有使用 shared。

四模式测试只采用 warmup=2/repeats=3/inner=10，用来检查接口契约。
其时间不能用于报告正式性能或宣称 O5/O6 已快于 O0。
仍须完成数据口径确认、24 样本同进程配对、正式 MSE、转换字节/吞吐和
性能瓶颈复盘。目标没有缩减为合成验证。

## 原始文件

- `runs/mixed_formats_validation_v3/validation.json`：逐项结果、合成 MSE、四模式原始时间、版本/二进制 hash。
- `runs/mixed_formats_validation_v3/codec_tests.txt`：完整单元测试输出。
- `reports/audit_mixed_formats_v2/audit.json`、`resources.txt`：同函数 ISA/资源审计。
- `reports/build_mixed_benchmark_v2.log`：CUDA 12.8 构建日志。
- `reports/mixed_formats_{memcheck,racecheck}_v3.log` 与对应 `_stdout.txt`：内存检查和该次验证脚本摘要。

完整 PTX/SASS 留在 A100 的
`/home/zlouyang/ADAngel_oyzl/reports/audit_mixed_formats_v2/`。
未在本文重复归档大型反汇编文本。

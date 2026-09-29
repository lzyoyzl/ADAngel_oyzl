# A100 O5–O10 分组实验

状态：六条路径已在 A100 编译运行，格式/数值、指令和小规模内存安全检查通过；24 样本四模式结果见 [分组报告](o5_o10_a100_results.md)。部分阶段 CV≥3%，全部原始值保留，不宣称严格稳定性全通过。不得将既有 O5/O6 历史结果直接作为本轮结果。

## 实验定义（命名版本 3）

| 组 | 后端 | 共同源格式，均从原始 FP16 独立准备 | 在线转换与计算 |
|---|---|---|---|
| A | O5 | W：NVFP4-G128；A：MXFP8 E4M3-G128 | 两者反量化到 FP16；HMMA，FP32 累加/输出 |
| A | O7 | 与 O5 **相同的编码张量** | W→Q4/F0，A→Q8/F−2；低 U4＋16×高 S4，两路 INT4 MMA |
| A | O9 | 与 O5 **相同的编码张量** | 同 O7 定点值；W 四个、A 八个补码 bitplane，32 个二进制组合 |
| B | O6 | W：HiF4-G128；A：实验变体 NV-style FP6 E2M3-G128 | 两者反量化到 FP16；HMMA，FP32 累加/输出 |
| B | O8 | 与 O6 **相同的编码张量** | W→Q4/F0，A→Q6/F2 后符号扩展到 INT8；两路 INT4 MMA |
| B | O10 | 与 O6 **相同的编码张量** | 同 O8 定点值；W 四个、A 六个補码 bitplane，24 个二进制组合 |

NVFP4 的 G128、HiF4 的 G128 扩展、FP6 的 E4M3/FP32 两级 scale 都是本实验已确认的格式选择，不冒充 NVIDIA 标准格式。细节见 `o5_o6_format_review.md`。MXFP8 转换后的有效 scale 乘 4，FP6 的有效 scale 除 4；权重 scale 不变。

旧命名 O5/O6 的双 INT4 路径现在是 O7/O8；历史日志保留原文，必须按其命名版本解释。O0–O4 和 SM120 运行入口不修改。

## 数据与公平性

原始数据目录：`data/raw/llama2_7b_prefill/`，已从 RTX 5090 授权复制到 A100。24 个 FP16 样本及 manifest 深度校验通过，manifest SHA256：`4ff05585d91f8940f20328b140c637ac948d0dbe47f1506db0cb5d839c9c3db0`。

每个样本只为每种源格式执行一次公共量化，组内三个后端复用相同张量，初始量化不计时。不从已有 INT8/MXFP4 反量化后再量化作为正式数据。已有 prepared trace 仅用于 O0/O1/O3 辅助参照及原始数据 provenance 对照。

主 MSE：O7/O9 相对 O5，O8/O10 相对 O6。另保留相对旧 O0 的辅助 MSE，不能混淆二者。CPU/Python 定点语义参考另行验证；同样定点数、组序和 FP32 FMA 顺序下，要求 O9 与 O7、O10 与 O8 输出逐位一致。

## 实现与计时

O5/O6 复用 O0 的 cuBLASLt 选择策略：FP16 输入、FP32 累加/输出、Tensor Core、无 split-K。只解码源格式，不经过定点舍入。

O7/O8 保留既有双 INT4 G128 优化。O9/O10 使用 `m16n8k128.b1.b1.and.popc`，最高补码平面为负位权。source 解码、RNE 定点转换、warp ballot packing、scale 补偿融合为一个转换 kernel，不生成中间 INT8 矩阵。GEMM 采用 `cp.async` 双缓冲、寄存器 partial、缓存权重 fragments、两条整数累加链、最后一次输出写回；A100 不使用 TMA。候选 CTA：64×64×128、64×128×256、64×64×512；扩大 pipeline 不合并 G128 scale。

计时保持双轨版本 2：转换阶段在 Event 内重复 100 次后摊销；compute/cold/steady 的 total 直接测一次执行，不以各阶段 median 相加。

这里的 Cold 指“不缓存本次权重转换”，不表示清空 GPU cache，也不包含文件读取、初始源量化、编译或显存申请。Steady-state 在每个测量计划的计时区间外准备权重，在重复执行期间复用它。

| 模式 | 内容 |
|---|---|
| conversion-only | 分别测 W、A 转换；total 为逐次摊销样本之和 |
| compute-only | 源数据已转换，仅 GEMM |
| cold | W 转换＋A 转换＋GEMM 单次直接计时 |
| steady-state | W 转换已缓存，A 转换＋GEMM 单次直接计时 |

所有申请、cuBLAS 选择、kernel 属性设置、校验和源量化均在计时外。预热 50 次、测量 200 次，轮换后端顺序；原始值、CV 和共享 GPU 状态全部保留，不删除离群点、不反复重测直到过线。不得把模拟格式或未通过审计的路径当作正式结果。

## 运行

本地提交后经 GitHub 同步 A100，再按原 SM80 环境重新编译。基础验证：

```bash
# 在 A100 项目目录和已有 adangel-a100 环境中执行；不安装/更换系统环境。
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps

python scripts/validate_a100_mixed_formats.py --output reports/mixed_formats_naming_v3
python scripts/validate_a100_mixed_trace_runner.py --original-fp16 --large \
  --output reports/mixed_trace_fixture_naming_v3
python scripts/audit_a100_o1.py --variant split_grouped --allow-spills \
  --output reports/mixed_int4_audit
python scripts/audit_a100_o1.py --variant mixed_binary --allow-spills \
  --output reports/mixed_binary_audit
compute-sanitizer --tool memcheck --error-exitcode 9 \
  python scripts/validate_a100_binary_safety.py --tiles 64x128x256 \
  --output reports/mixed_binary_memcheck.json
compute-sanitizer --tool racecheck --error-exitcode 9 \
  python scripts/validate_a100_binary_safety.py --tiles 64x128x256 \
  --output reports/mixed_binary_racecheck.json
```

`--allow-spills` 只把已批准的小量 local/stack 资源作为警告保留，不跳过原生 ISA 检查。
新的输出目录/文件须不存在；重新测试时改名，保留原始结果。

正式运行（目录须不存在；先用 `--samples 1 --rounds 1 --warmup 5 --repeats 20` 冒烟）：

```bash
python scripts/benchmark_a100_mixed_trace.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --raw-data data/raw/llama2_7b_prefill \
  --output runs/a100_o5_o10_v3 \
  --scale-layouts group_major --binary-scale-layouts row_major \
  --binary-tile 64x128x256 \
  --samples 24 --rounds 1 --warmup 50 --repeats 200 --inner 100

python scripts/report_a100_o5_o10.py \
  --input runs/a100_o5_o10_v3 --output reports/o5_o10_grouped_results.md
```

候选 tile 不代表已胜出；正式选择应根据同输入配对性能、MSE、SASS 和 sanitizer 验收。报告须分别列出两组的四种计时、组内加速比和输出 MSE，不承诺 Binary 必然胜过 FP16。

`--modes compute_only` 可用于单独的多轮复测，默认仍运行四种模式。复测必须使用新目录、记录轮数并单独展示，不能静默覆盖首次记录或混成同一轮结果。报告脚本会拒绝把这种仅计算的复测误标为四模式结果。

格式/指令/内存安全证据见 [v3 preflight](evidence/a100_o5_o10_preflight_v3/README.md)；
实现细节见 [kernel 工作流](mixed_precision_kernel_workflow.md)。保留完整 stage、原始 Event 样本和环境数据，主文表格按组列 median/mean、配对加速比与 MSE。

# O1/O2/O3/O4 Nsight Compute profiling

本页规定 RTX 5090 上的 NCU 采集接口。正式延迟仍以 CUDA Event 实验为准；NCU 的
kernel replay、cache control、clock control 和软件 patch 会改变运行时间，因此报告中的
Duration 只用于同一采集设置下的诊断。

## 1. 专用驱动

`scripts/profile_ncu_kernel.py` 只加载一个已经校验的真实 prepared 样本，只运行指定
production variant 的 `compute_only` 路径，不执行 Python reference，也不运行其他
计时模式。默认样本是 `layer_00_q_proj`，形状必须为 `4096×4096×4096`。

先确认接口：

```bash
python scripts/profile_ncu_kernel.py --help
```

不使用 NCU 的冒烟命令：

```bash
python scripts/profile_ncu_kernel.py \
  --data data/prepared/llama2_7b_prefill \
  --sample-id layer_00_q_proj \
  --variant o3 \
  --warmup 5 --repeats 1
```

驱动会在计时前完成该 variant 的格式准备，随后发射5个warmup production GEMM和1个
measured production GEMM。因此NCU必须使用：

```text
--launch-skip 5 --launch-count 1
```

从而只采集第6个匹配的production kernel。

## 2. Production kernel过滤器

| Variant | NCU `--kernel-name` |
|---|---|
| O1 | `regex:adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup` |
| O2 | `regex:cutlass::device_kernel` |
| O3 | `regex:adangel_o3_split_tma_ws` |
| O4 | `regex:adangel_o4_bitwise_tma_ws` |

O2驱动在目标launch之前不执行其他CUTLASS GEMM，因此该过滤器只会匹配正式O2
MXFP4 kernel。每次报告还必须检查NCU显示的kernel name与驱动输出的`kernel_symbol`。

## 3. 分层采集

所有报告保存到 `reports/ncu/`，该目录只保存服务器生成的诊断产物，不提交大型
`.ncu-rep` 文件。

基础报告：

```bash
ncu \
  --set basic \
  --kernel-name '<VARIANT_FILTER>' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none \
  --export reports/ncu/<variant>_basic \
  --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill \
    --sample-id layer_00_q_proj \
    --variant <variant> --warmup 5 --repeats 1
```

计算、调度和源码报告：

```bash
ncu \
  --section ComputeWorkloadAnalysis \
  --section InstructionStats \
  --section SchedulerStats \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name '<VARIANT_FILTER>' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none \
  --export reports/ncu/<variant>_compute \
  --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill \
    --sample-id layer_00_q_proj \
    --variant <variant> --warmup 5 --repeats 1
```

内存报告：

```bash
ncu \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Tables \
  --section LaunchStats \
  --section Occupancy \
  --kernel-name '<VARIANT_FILTER>' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none \
  --export reports/ncu/<variant>_memory \
  --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill \
    --sample-id layer_00_q_proj \
    --variant <variant> --warmup 5 --repeats 1
```

不要一开始使用`--set full`。它在NCU 2025.1.1中需要采集约5895个metric，会产生
大量replay；只有分层报告仍无法解释瓶颈时才使用。

## 4. 可比性规则

- O1/O2/O3/O4必须使用同一个真实样本、`4096³`、相同warmup和相同NCU section；
- 每次只采集一个production launch；
- 使用`--clock-control none`，记录动态Boost条件；
- 第一轮保留NCU默认cache flush，分析隔离kernel；
- 如需研究steady-state cache，再额外使用确定性application replay与
  `--cache-control none`，不得与默认报告混合比较；
- NCU Duration不得写入正式CUDA Event性能表；
- 报告必须保留原始`.ncu-rep`、命令、NCU版本和导出的关键指标。

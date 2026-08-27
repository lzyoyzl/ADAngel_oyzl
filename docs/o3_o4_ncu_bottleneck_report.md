# O3/O4 Nsight Compute 性能瓶颈分析报告

## 1. 结论摘要

本报告使用 RTX 5090 上的 Nsight Compute（NCU）对 O1/O2/O3/O4 的 production
GEMM kernel 进行同条件对照。采集对象固定为真实 trace
`layer_00_q_proj`，矩阵形状固定为 `M=N=K=4096`，只分析 `compute_only` 路径。

结论如下：

1. **O3 不是 DRAM、TMA 或 barrier 受限，而是 INT4 操作数准备和整数指令开销受限。**
   O3 的 SM throughput 为 `91.37%`，DRAM throughput 仅为 `1.62%`；动态 warp
   指令数约为 O1 的 `6.11×`。SASS 中约 `81.48%` 的动态指令属于整数 ALU，主要是
   `LOP3`、`SHF` 和 `IMAD`。这说明 low/high 两类 INT4 MMA 路径本身不是全部成本，CuTe sub-byte
   fragment 的提取、重排、地址计算和软件重构占据了大量执行资源。
2. **O4 是 8×4 bitplane 展开的计算与片上数据通路联合瓶颈。** O4 每个 G128
   必须计算 `8×4=32` 个 plane pair。其 SM throughput 为 `94.25%`，DRAM throughput
   仅为 `0.47%`，动态 warp 指令数约为 O1 的 `19.40×`；同时产生约 `1.894B`
   shared-memory wavefront。它不是“没有用上 Tensor Core”，而是精确 8×4 语义本身
   需要大量 BMMA、bitplane fragment 搬运和带符号加权重构。
3. **warm-cache 复测不改变结论。** `--cache-control none` 下 O3/O4 的 Duration、SM、
   L1/L2 和 DRAM 指标与默认 cache flush 基本一致，因此性能差异不能归因于 NCU
   刷新 cache。
4. **O3 有明确的继续优化空间，O4 在不改变 8×4 精确语义时很难达到 O1。** O3
   应优先减少通用 sub-byte fragment 生成的 `LOP3/SHF/IMAD`，其次增大 N tile 和修复
   shared-memory bank conflict。O4 应优先减少 32 个 plane pair 周围的寄存器重排、
   accumulator 更新和 shared-memory 流量；但不能通过优化消除 32 个逻辑 BMMA 的
   算法下界。

## 2. 实验环境与边界

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，SM120 |
| Driver | 595.58.03 |
| CUDA Toolkit | 12.8，nvcc 12.8.93 |
| Nsight Compute | 2025.1.1.0，build 35528883 |
| 数据 | `data/prepared/llama2_7b_prefill_o0_o4/layer_00_q_proj.pt` |
| 形状 | `M=N=K=4096` |
| 模式 | `compute_only` |
| 预热/采集 | 5 次 warmup，采集第 6 次 production launch |
| 时钟 | `--clock-control none`，不锁频 |
| 第一轮 cache | `--cache-control all`，隔离单次 kernel |
| 复核 cache | application replay + `--cache-control none` |

NCU 的 replay、patch 和计数器采集会改变运行时间。本文中的 NCU Duration 只用于同一
NCU 设置下的诊断，不替代 24 样本正式实验的 CUDA Event 延迟。

## 3. 被采集的正式 kernel

| Variant | Production kernel | Tile / 主要语义 |
|---|---|---|
| O1 | `adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup` | `128×64×64`，两个 K32/group stage，INT8 IMMA，软件 scale |
| O2 | CUTLASS `device_kernel` | `128×128×256`，原生 MXFP4 block-scaled MMA |
| O3 | `adangel_o3_split_tma_ws<O3Config<16,1,false>>` | `128×16×128`，A8→U4 low/S4 high；每个 G128 各执行两个 K64 low/high atom，共 4 次 MMA atom 调用 |
| O4 | `adangel_o4_bitwise_tma_ws<O4Config<64,64,4,4,2,true>>` | `64×64×512`，8×4 bitplane，32 BMMA/G128 |

四者均使用 TMA、cooperative warp specialization、寄存器 partial、FP32 累加/输出，
最终输出元素只写回一次。本报告没有把 probe、旧 baseline 或消融候选混入采样。

## 4. 正式 CUDA Event 性能背景

下表来自 24 个真实样本的正式 `compute_only` 结果，表中数值是“24 个逐样本
median latency 的 median”，不是 NCU Duration。

| Variant | 正式 median latency (ms) | 相对 O1 延迟 | 相对 O2 延迟 |
|---|---:|---:|---:|
| O1 | 0.622136 | 1.000× | 4.492× |
| O2 | 0.138512 | 0.223× | 1.000× |
| O3 | 2.223352 | 3.574× | 16.052× |
| O4 | 7.632488 | 12.268× | 55.103× |

后续 NCU 只负责解释这个差距从哪里来。

## 5. NCU 基础指标

### 5.1 默认 cache flush

| 指标 | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| NCU Duration (ms) | 0.621504 | 0.115520 | 2.060736 | 7.232224 |
| SM throughput (%) | 41.04 | 64.88 | 91.37 | 94.25 |
| DRAM throughput (%) | 6.21 | 23.10 | 1.62 | 0.47 |
| L2 throughput (%) | 80.40 | 80.03 | 30.96 | 3.19 |
| L1 active throughput (%) | 65.85 | 44.13 | 35.62 | 97.76 |
| Registers/thread（分配） | 56 | 168 | 56 | 96 |
| Dynamic shared memory (KiB) | 37.63 | 94.00 | 34.25 | 98.13 |
| Achieved warp occupancy (%) | 70.41 | 20.68 | 69.82 | 35.21 |
| Waves/SM | 6.02 | 6.02 | 24.09 | 24.09 |

解读：

- O3/O4 的 SM 已经很忙，而 DRAM 几乎空闲，排除全局显存带宽瓶颈。
- O2 虽然 occupancy 低，但原生 MXFP4 MMA 每条指令完成的有效工作多，低 occupancy
  并没有阻止它成为最快实现。
- O4 的 `100480 B` dynamic shared memory 和 96 个分配寄存器使每个 SM 最多驻留一个
  CTA。它降低延迟隐藏能力，但 O4 的 scheduler 仍能持续发射，因此这是次要放大因素，
  不是 12.27× 差距的唯一原因。

### 5.2 warm-cache 复核

| 指标 | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| NCU Duration (ms) | 0.626944 | 0.118144 | 2.069664 | 7.246272 |
| SM throughput (%) | 40.98 | 64.16 | 91.34 | 94.22 |
| DRAM throughput (%) | 5.99 | 31.13 | 1.92 | 0.55 |
| L2 throughput (%) | 79.91 | 78.79 | 30.82 | 3.19 |
| L1 active throughput (%) | 64.79 | 43.46 | 35.61 | 97.75 |

O3/O4 的 Duration 变化均小于 1%，SM/DRAM 结论不变。因此后续分析采用第一轮统一
cache flush 的 compute/memory 报告，不混用两种 cache 设置的 metric。

## 6. 指令与调度证据

### 6.1 动态指令规模

| 指标 | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| Warp instructions（raw） | 454.35M | 38.93M | 2.777B | 8.815B |
| 相对 O1 | 1.00× | 0.086× | 6.11× | 19.40× |
| Tensor pipeline active (%) | 24.60 | 71.31 | 13.53 | 62.95 |
| Integer/ALU pipeline active (%) | 14.37 | 5.23 | 48.95 | 39.60 |
| LSU pipeline active (%) | 43.21 | 36.95 | 18.29 | 97.75 |
| TMA pipeline active (%) | 0.58 | 0.53 | 0.24 | 0.01 |

各 pipeline 百分比是相对各自 peak 的 active 指标，不能相加为 100%。关键观察是：

- O3 的总指令量已经远大于“只比 O1 多一条 INT4 MMA”所能解释的程度；
- O4 的 32 plane pair 使 Tensor/LSU 与片上 fragment 处理同时繁忙；
- TMA active 很低不代表未使用 TMA，而是 TMA 事务不是 limiting pipeline。

### 6.2 SASS 动态分类

| Variant | 主要动态指令证据 |
|---|---|
| O1 | 总计约 452.73M；FP 135.3M、整数 ALU 116.7M、MMA 16.78M、shared 21.5M |
| O2 | 总计约 38.37M；原生 `OMMA` 8.39M、shared 7.88M、整数 ALU 7.84M |
| O3 | 总计约 2.798B；整数 ALU 2.280B（81.48%），其中 `LOP3` 1.079B、`SHF` 536.9M、`IMAD` 402.7M；两类 INT4 MMA 合计 33.55M |
| O4 | 总计约 8.829B；整数 ALU 3.391B、NCU 归类为 other 的 MOV/未解码指令 4.664B、可识别 IMMA-class 指令 536.87M |

O3 的 33.55M INT4 MMA 体现 low/high 两类计算路径；每条逻辑 G128 路径由两个 K64
atom 覆盖，但真正占主导的是其周围的
packed nibble 提取、lane layout 重排和地址生成。O4 可识别的相关 MMA-class 动态数约为
O1 MMA 的 32×，与 8×4 plane pair 的结构吻合。

### 6.3 Scheduler

| 指标 | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| Issue active (%) | 42.40 | 20.85 | 70.09 | 64.75 |
| Active warps/scheduler | 8.238 | 2.479 | 8.377 | 4.234 |
| Eligible warps/scheduler | 1.355 | 0.269 | 3.647 | 1.108 |
| 平均 cycles/instruction | 19.43 | 11.89 | 11.95 | 6.54 |
| Barrier stall / issued inst | 0.015 | 0.350 | 0.052 | 0.001 |
| Long-scoreboard stall / issued inst | 1.824 | 0.597 | 0.154 | 0.008 |
| Math-pipe-throttle / issued inst | 0.555 | 2.497 | 3.935 | 0.543 |

O3/O4 都不是 barrier 或 long-scoreboard 主导。O3 同时有较多 eligible warp、较高 issue
active 和较高 math-pipe throttle，符合“大量整数操作持续占用执行管线”的特征。O4
occupancy 较低，但仍能保持 64.75% issue active；其根因是必须发射的工作太多，而不是
大多数周期完全无 warp 可发射。

## 7. 内存与片上搬运证据

| 指标 | O1 | O2 | O3 | O4 |
|---|---:|---:|---:|---:|
| DRAM read (MiB) | 32.53 | 17.05 | 24.16 | 24.20 |
| DRAM write (MiB) | 32.20 | 27.62 | 32.20 | 32.17 |
| L2 hit rate (%) | 97.13 | 86.59 | 99.08 | 96.91 |
| L1 hit rate (%) | 91.74 | 99.98 | 83.16 | 93.34 |
| Shared wavefronts | 149.44M | 20.25M | 345.89M | 1.894B |
| Shared bank conflicts | 53.83M | 1.70M | 251.79M | 223.76M |
| Conflicts / wavefronts | 0.360 | 0.084 | 0.728 | 0.118 |

`shared wavefronts` 是 NCU 的片上事务计数，不是字节数。O3 的 bank conflict 比例很高，
说明 shared layout/swizzle 仍有优化价值；但其 shared pipeline 未达到峰值，且整数 ALU
占比更高，所以 bank conflict 是第二优先级。O4 的 shared wavefront 是 O1 的约 12.68×、
O3 的约 5.48×，与反复提供 bitplane fragment 的实现结构一致。

## 8. O3 为什么慢于 O1/O2

O3 每个 G128 的数学路径是：

```text
A8 = A_low_u4 + 16 * A_high_s4

P_group = MMA_u4s4(A_low, W_q4)
        + 16 * MMA_s4s4(A_high, W_q4)
```

因此即使忽略全部软件开销，O3 也至少需要 low/high 两条逻辑 INT4 Tensor Core 路径；
由于硬件 atom 是 K64，每个 G128 上 low/high 各调用两次，共 4 次 MMA atom 调用。实际
实现还需要：

1. 从 packed A/W shared tile 中提取 4-bit fragment；
2. 按 MMA lane layout 做寄存器重排；
3. 维护 U4/S4 两种输入语义；
4. 执行 `low + 16*high` 重构；
5. 每个 G128 解码并应用 scale；
6. 使用 `N=16` tile，产生 `256×32=8192` 个 CTA，而 O1 只有 `64×32=2048`
   个 CTA，重复了更多 CTA 级地址和 fragment 准备工作。

NCU 显示 O3 的整数 ALU 指令达到 2.280B，而两种 INT4 MMA 合计仅 33.55M。故当前
O3 的首要问题不是“INT4 Tensor Core 算得慢”，而是 Tensor Core 前后的通用 sub-byte
数据组织成本过高。

O2 则不同：原生 SM120 MXFP4 block-scaled MMA 在一条硬件语义中直接消费 E2M1 和
UE8M0 scale，CTA tile 为 `128×128×256`，不需要 O3 的 low/high 软件拆分与重构。因此
“INT4 位宽是 MXFP4 的一半，所以 O3 应该是 O2 的一半性能”并不成立。

## 9. O4 为什么慢于 O1/O2

O4 对每个 G128 执行：

```text
A8  -> 8 个二进制 plane，权重 [1,2,4,8,16,32,64,-128]
WQ4 -> 4 个二进制 plane，权重 [1,2,4,-8]

dot(A8, WQ4)
  = sum_{i=0..7} sum_{j=0..3}
      a_weight[i] * w_weight[j] * BMMA(A_plane[i], W_plane[j])
```

这意味着每个组有 32 个不可随意删除的 plane pair。当前实现已经采用：

- TMA + cooperative warp specialization；
- `K512` pipeline，一次覆盖四个 G128；
- 两条独立 BMMA accumulator chain；
- B fragment cache；
- 寄存器 partial，最终只写一次输出。

但这些优化只能减少调度、搬运和依赖开销，不能把精确 8×4 语义的 32 个逻辑 BMMA
变成 1 个。NCU 进一步显示热点位于 plane loop 周围的
`group_accumulators += coefficient * d`、A/B fragment 地址与加载、MOV/LOP3/SHF 和
未解码的 SM120 binary 指令。O4 同时被 100480 B shared memory 限制为一个 CTA/SM。

因此 O4 比 O1 慢 12.27× 是“32 倍逻辑二进制乘积经过较高 BMMA 吞吐部分抵消后”的
结果。若保持完整 8×4 two's-complement 精确重构，要求它达到 O1 的延迟并不现实；若
允许论文中的 selective fusion、截断或减少 bitplane，则属于改变算法语义，必须作为新
variant 单独报告，不能冒充当前 O4。

## 10. 后续优化优先级

### 10.1 O3

1. **最高优先级：替换通用 sub-byte fragment 展开。** 使用经过 lane/layout 单元测试
   验证的 packed register load、专用 copy 或 inline PTX，减少 `LOP3/SHF/IMAD`。
2. **复用解包后的 fragment。** 让同一个 A/W packed fragment 服务更多 output replica，
   避免消费者重复提取 nibble。
3. **扩大 N tile。** 在完成第 1 项后消融 `N=32/64`，减少 8192 个 CTA 带来的重复
   地址/fragment 准备。必须同时检查寄存器 spill、shared memory 和 occupancy。
4. **修复 shared bank conflict。** 针对 A/W shared layout 加 swizzle 或改变访问排列，
   以 `251.79M` bank conflict 为回归指标。
5. 双 G128 pipeline、指令级双缓冲只在操作数准备开销下降后再评估；当前根因并不是
   TMA pipeline 等待。

### 10.2 O4

1. **减少 plane loop 周围的寄存器搬运和 accumulator 更新。** 将系数符号与 2 的幂
   权重合并进更少的 shift/add 路径，同时严格验证 INT32 范围和 FP32 累加顺序。
2. **进一步复用 A/B bitplane fragment。** B 已缓存，下一步重点检查 A plane 是否能跨
   replica/chain 复用，并减少 MOV/LOP3/SHF。
3. **降低 shared-memory footprint。** 只有降到约 50 KiB/CTA 且寄存器也允许时，才可能
   从一个 CTA/SM 提升到两个；需要重新消融 K256/K512，而不是仅凭 occupancy 推断。
4. **谨慎增加 BMMA 依赖链。** 当前已有两条 chain，且 barrier/scoreboard stall 很低；
   继续增加 chain 可能只会增加寄存器压力，必须用 NCU A/B 证据决定。
5. 在做单条 binary instruction 级优化前，优先使用支持 SM120 完整反汇编的更新版
   Nsight Compute/CUDA 工具复核 `???` 指令。

每次优化后应使用同一真实样本和相同 NCU sections 重新采集，并最终回到 24 样本
CUDA Event、MSE、instruction audit 和 CV 验收。单个 NCU 样本不能替代正式统计。

## 11. 工具限制与审计关系

NCU 2025.1.1 对部分 SM120 binary SASS 显示为 `???`。在 O4 source report 中，约
1.611B 动态指令落入这一类别，因此不能仅依赖 NCU 的 opcode 文本判断是否存在 BMMA。

当前结论使用三层证据：

1. production 元数据明确指定 `SM80_16x8x128_S32U1U1S32_TN_ANDPOPC`；
2. 既有 PTX/SASS 指令审计确认正式 O4 symbol 中存在 binary MMA 路径；
3. NCU 的动态计数、source hotspot 与 32 plane pair 结构一致。

所以“NCU 显示部分 `???`”是反汇编支持限制，不代表 O4 退化为 CUDA Core 软件点积。

## 12. 实际执行的 NCU 命令

以下命令均在服务器项目目录 `~/oyzl/ADAngel_oyzl` 内执行。为避免重复四段完全相同的
长命令，compute/memory 小节把实际逐条执行的四个 variant 无损写成 shell loop；loop
中的每个 `ncu` invocation、filter、section 和参数均与实际采集一致。

### 12.1 环境与权限

```bash
/usr/local/cuda-12.8/bin/ncu --version
/usr/local/cuda-12.8/bin/ncu --list-sets
/usr/local/cuda-12.8/bin/ncu --list-sections

/usr/local/cuda-12.8/bin/ncu \
  --set basic \
  --kernel-name 'regex:adangel_o3_split_tma_ws' \
  --launch-count 1 --clock-control none \
  --export reports/ncu/permission_o3_basic --force-overwrite \
  python scripts/validate_o3.py \
    --m 128 --n 128 --k 128 --warmup 0 --repeats 1 \
    --implementation production
```

最后一条用最小 O3 production launch 验证当前用户拥有 performance counter 权限，
并确认 NCU 可连接目标进程。

### 12.2 Basic 报告

四个 variant 分别执行以下命令，只替换 filter、输出名和 variant：

```bash
/usr/local/cuda-12.8/bin/ncu --set basic \
  --kernel-name 'regex:^adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup$' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control all \
  --export reports/ncu/o1_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o1 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic \
  --kernel-name 'regex:^device_kernel$' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control all \
  --export reports/ncu/o2_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o2 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic \
  --kernel-name 'regex:^adangel_o3_split_tma_ws' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control all \
  --export reports/ncu/o3_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o3 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic \
  --kernel-name 'regex:^adangel_o4_bitwise_tma_ws' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control all \
  --export reports/ncu/o4_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o4 --warmup 5 --repeats 1
```

最初曾用 `--kernel-name 'regex:cutlass::device_kernel'` 采集 O2，但 NCU 报告
`No kernels were profiled`；NCU 2025.1.1 实际暴露的简化名称是 `device_kernel`。随后改为
严格过滤器 `regex:^device_kernel$` 并成功采集。驱动和文档已同步修正。

离线导出命令：

```bash
for variant in o1 o2 o3 o4; do
  /usr/local/cuda-12.8/bin/ncu \
    --import reports/ncu/${variant}_basic.ncu-rep \
    --page raw --csv > reports/ncu/${variant}_basic_raw.csv
done
```

### 12.3 Compute、scheduler 与 source 报告

以下循环对 O1/O2/O3/O4 各执行一次：

```bash
for spec in \
  'o1|regex:^adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup$' \
  'o2|regex:^device_kernel$' \
  'o3|regex:^adangel_o3_split_tma_ws' \
  'o4|regex:^adangel_o4_bitwise_tma_ws'; do
  IFS='|' read -r VARIANT FILTER <<< "${spec}"
  /usr/local/cuda-12.8/bin/ncu \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section SchedulerStats \
    --section WarpStateStats \
    --section SourceCounters \
    --kernel-name "${FILTER}" \
    --launch-skip 5 --launch-count 1 \
    --clock-control none --cache-control all \
    --export reports/ncu/${VARIANT}_compute --force-overwrite \
    python scripts/profile_ncu_kernel.py \
      --data data/prepared/llama2_7b_prefill_o0_o4 \
      --sample-id layer_00_q_proj --variant "${VARIANT}" \
      --warmup 5 --repeats 1
done
```

每个报告需要 22 个 kernel replay pass。随后执行：

```bash
for variant in o1 o2 o3 o4; do
  /usr/local/cuda-12.8/bin/ncu \
    --import reports/ncu/${variant}_compute.ncu-rep \
    --page raw --csv > reports/ncu/${variant}_compute_raw.csv

  /usr/local/cuda-12.8/bin/ncu \
    --import reports/ncu/${variant}_compute.ncu-rep \
    --page source --print-source sass --csv \
    > reports/ncu/${variant}_source_sass.csv
done

/usr/local/cuda-12.8/bin/ncu --help | \
  grep -A8 -B3 -E 'page|print-source|source-fold' | head -120

/usr/local/cuda-12.8/bin/ncu \
  --import reports/ncu/o3_compute.ncu-rep \
  --page source --print-source cuda,sass --csv \
  > reports/ncu/o3_source_cuda_sass.csv

/usr/local/cuda-12.8/bin/ncu \
  --import reports/ncu/o4_compute.ncu-rep \
  --page source --print-source cuda,sass --csv \
  > reports/ncu/o4_source_cuda_sass.csv
```

`--page source` 用于把热点映射到 SASS/源码；源码行相关性会受到 header inline、模板和
优化影响，不能把错误关联到 host helper 的行号解释成该 host 代码在 GPU 上运行。

### 12.4 Memory、launch 与 occupancy 报告

以下循环同样对四个 variant 各执行一次：

```bash
for spec in \
  'o1|regex:^adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup$' \
  'o2|regex:^device_kernel$' \
  'o3|regex:^adangel_o3_split_tma_ws' \
  'o4|regex:^adangel_o4_bitwise_tma_ws'; do
  IFS='|' read -r VARIANT FILTER <<< "${spec}"
  /usr/local/cuda-12.8/bin/ncu \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section LaunchStats \
    --section Occupancy \
    --kernel-name "${FILTER}" \
    --launch-skip 5 --launch-count 1 \
    --clock-control none --cache-control all \
    --export reports/ncu/${VARIANT}_memory --force-overwrite \
    python scripts/profile_ncu_kernel.py \
      --data data/prepared/llama2_7b_prefill_o0_o4 \
      --sample-id layer_00_q_proj --variant "${VARIANT}" \
      --warmup 5 --repeats 1
done
```

每个报告需要 31 个 replay pass。离线导出：

```bash
for variant in o1 o2 o3 o4; do
  /usr/local/cuda-12.8/bin/ncu \
    --import reports/ncu/${variant}_memory.ncu-rep \
    --page raw --csv > reports/ncu/${variant}_memory_raw.csv
done
```

### 12.5 warm-cache application replay

四个 variant 分别执行：

```bash
/usr/local/cuda-12.8/bin/ncu --set basic --replay-mode application \
  --kernel-name 'regex:^adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup$' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control none \
  --export reports/ncu/o1_steady_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o1 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic --replay-mode application \
  --kernel-name 'regex:^device_kernel$' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control none \
  --export reports/ncu/o2_steady_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o2 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic --replay-mode application \
  --kernel-name 'regex:^adangel_o3_split_tma_ws' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control none \
  --export reports/ncu/o3_steady_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o3 --warmup 5 --repeats 1

/usr/local/cuda-12.8/bin/ncu --set basic --replay-mode application \
  --kernel-name 'regex:^adangel_o4_bitwise_tma_ws' \
  --launch-skip 5 --launch-count 1 \
  --clock-control none --cache-control none \
  --export reports/ncu/o4_steady_basic --force-overwrite \
  python scripts/profile_ncu_kernel.py \
    --data data/prepared/llama2_7b_prefill_o0_o4 \
    --sample-id layer_00_q_proj --variant o4 --warmup 5 --repeats 1
```

离线导出：

```bash
for variant in o1 o2 o3 o4; do
  /usr/local/cuda-12.8/bin/ncu \
    --import reports/ncu/${variant}_steady_basic.ncu-rep \
    --page raw --csv > reports/ncu/${variant}_steady_basic_raw.csv
done
```

application replay 会为各 metric pass 重新运行整个 Python 程序，因此程序打印的 CUDA
Event 时间会被 NCU 注入放大到数百毫秒甚至数秒。它们不是正式结果；本节只读取合并后
`.ncu-rep` 中的硬件计数器。

## 13. 最终判断

- **O3：实现瓶颈为主。** low/high 两类 INT4 路径的算法成本不可避免，但当前 2.280B 整数
  ALU 指令和高 bank conflict 表明，operand preparation/layout 仍有显著优化空间。
- **O4：算法展开成本为主，片上实现成本为辅。** 32 plane pair 是保持完整 8×4 精确
  语义的下界；shared/register/layout 仍可优化，但很难在不改变语义时达到 O1。
- **O1/O2 快不是因为测量口径不同。** 四者采集同一个真实样本、同一 `compute_only`
  范围和同一 NCU section；O2 依靠原生 block-scaled MMA，O1 只需每 K32 一条 INT8
  partial 路径，而 O3/O4 需要软件拆分与多路径重构。

报告结论与当前 instruction audit、MSE 验收及 24 样本正式性能相容，没有发现 kernel
退化、错误匹配或 DRAM bottleneck 的证据。

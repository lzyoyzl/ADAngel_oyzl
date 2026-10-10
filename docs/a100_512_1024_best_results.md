# A100 512³ 与 1024³ 当前最佳后端实验

## 实验结论

在 A100 上完成 O1/O3/O5/O6/O7/O8 的实际 `M=N=K=512` 和 `1024` 测试，不补齐到4096，也不切换到旧的兼容实现。

1024³ 下，O3 的 GEMM 相对 O1 约快 **2.06×**，Cold 约快 **1.48×**；O7/O8 的 GEMM 相对 O5/O6 约快 **1.21×**，但转换开销抵消了大部分端到端收益。512³ 下，O7/O8 与 FP16 基线的 GEMM 中位数相同，Cold 反而更慢。当前4096最佳实现的优势不能直接外推到小矩阵。

## 实验方法与实现

使用 A100-PCIE-40GB，108个SM，PyTorch 2.7.1+cu128、CUDA 12.8。复用24个 Llama-2-7B 原始 FP16 trace，先取 `A[:S,:S]` 和 `W[:S,:S]`，再分别使用原有量化方法生成源格式，计算 `Y=A@W.T`。

这些是原4096-token trace 的子矩阵，**不是重新运行512或1024-token模型前向**。公共 FP16→源格式准备不计时；实际定点转换、反量化及 factor/guard 等 metadata 准备计入对应转换阶段。

| 后端 | 本轮执行的正式实现 | CTA 与分组 |
|---|---|---|
| O1 | `swizzle_128x64_k128_magic`，INT8 MMA、寄存器 partial、精确 magic-bias 转换 | 128×64×128；软件 G32 scale |
| O3 | v89 全K安全整数累加，八链调度、M8分组CTA，两路原生 INT4 | 64×128×128；G128 |
| O5 | NVFP4/MXFP8 反量化到FP16，cuBLASLt FP16 Tensor Core | 库选择 tile；源格式 G128 |
| O6 | HiF4/实验变体FP6 反量化到FP16，cuBLASLt FP16 Tensor Core | 库选择 tile；源格式 G128 |
| O7 | v99 全K安全整数累加、streaming输出；v139转换组合 | 64×128×128；G128 |
| O8 | 同一v99；v138转换组合 | 64×128×128；G128 |

O3/O7/O8 只将已应用算法实例化到 K512/K1024，G128 组数由32变为4/8；转换器增加有效组边界检查，不做新的 tile 搜索或小矩阵算法调优。两路指令仍为 `U4×S4` 与 `S4×S4`。整数安全检查不通过时，保留逐组FP32缩放的安全路径；本轮24个真实样本均未触发该回退。

每个样本、尺寸、后端和模式测3轮，每轮预热1000次、正式200次；转换阶段每个 Event 区间重复100次并摊销。单 CUDA stream、预分配内存，交错后端顺序，不锁频，不要求GPU空闲。主实验共 **3,456条记录**，保留全部原始观测。

## 四种计时结果

全部延迟单位为 **µs**，`1 µs = 0.001 ms`。Median：先取每轮200次的中位数，再取每样本3轮中位数，最后汇总24样本中位数；Mean：对应阶段全部24×3×200次观测的平均值，不删除离群值。

### Conversion-only

W为权重转换或反量化，A为激活转换或反量化。O1不转换激活，因此为“—”。Total为各独立摊销阶段同编号样本之和；其统计中位数不保证等于两列中位数直接相加。

| 尺寸 | 后端 | W Median µs | A Median µs | Total Median µs | Total Mean µs |
|---|---|---:|---:|---:|---:|
| 512³ | O1 | 3.707 | — | 3.707 | 3.767 |
| 512³ | O3 | 6.589 | 3.246 | 9.838 | 9.965 |
| 512³ | O5 | 3.983 | 3.738 | 7.716 | 7.743 |
| 512³ | O6 | 3.942 | 3.738 | 7.706 | 7.866 |
| 512³ | O7 | 4.475 | 8.305 | 12.780 | 12.785 |
| 512³ | O8 | 4.483 | 8.279 | 12.764 | 12.784 |
| 1024³ | O1 | 6.595 | — | 6.595 | 6.607 |
| 1024³ | O3 | 7.020 | 3.717 | 10.737 | 10.812 |
| 1024³ | O5 | 7.291 | 6.298 | 13.588 | 13.592 |
| 1024³ | O6 | 7.557 | 6.932 | 14.487 | 14.484 |
| 1024³ | O7 | 5.990 | 10.624 | 16.612 | 16.613 |
| 1024³ | O8 | 5.724 | 10.245 | 15.974 | 15.975 |

### Compute-only GEMM

转换后的操作数及 metadata 已准备，只计GEMM。此处 GEMM-only 与 Compute-only 含义相同。

| 后端 | 512³ Median µs | 512³ Mean µs | 1024³ Median µs | 1024³ Mean µs |
|---|---:|---:|---:|---:|
| O1 | 15.360 | 15.876 | 37.888 | 37.605 |
| O3 | 10.240 | 10.642 | 18.432 | 17.997 |
| O5 | 10.240 | 10.367 | 23.552 | 23.111 |
| O6 | 10.240 | 10.503 | 23.552 | 23.132 |
| O7 | 10.240 | 10.362 | 19.456 | 19.111 |
| O8 | 10.240 | 10.356 | 19.456 | 19.116 |

### Cold total

每次重新执行W转换、A转换和GEMM，直接计单次端到端间隔。不含源格式公共量化、内存分配、文件读取或模型加载；不是进程或缓存冷启动。

| 后端 | 512³ Median µs | 512³ Mean µs | 1024³ Median µs | 1024³ Mean µs |
|---|---:|---:|---:|---:|
| O1 | 22.528 | 22.433 | 47.104 | 47.040 |
| O3 | 22.528 | 23.222 | 31.744 | 31.549 |
| O5 | 20.480 | 20.958 | 38.912 | 39.354 |
| O6 | 20.480 | 21.360 | 39.936 | 40.404 |
| O7 | 26.624 | 27.033 | 38.912 | 38.749 |
| O8 | 25.600 | 26.918 | 37.888 | 38.071 |

### Steady-state total

缓存W转换结果和权重metadata，计在线A处理加GEMM。O1保持INT8激活不变，因此仅剩GEMM。

| 后端 | 512³ Median µs | 512³ Mean µs | 1024³ Median µs | 1024³ Mean µs |
|---|---:|---:|---:|---:|
| O1 | 15.872 | 15.904 | 37.888 | 37.598 |
| O3 | 16.384 | 16.801 | 24.576 | 24.639 |
| O5 | 16.384 | 16.679 | 32.768 | 32.339 |
| O6 | 16.384 | 17.675 | 32.768 | 32.812 |
| O7 | 21.504 | 22.318 | 32.768 | 32.609 |
| O8 | 21.504 | 21.680 | 31.744 | 32.180 |

Cold/Steady 是独立单次测量，不能用 Conversion-only 与 Compute-only 的统计值相加代替。

## 输出 MSE

MSE对最终FP32输出矩阵逐元素计算，使用FP64 reduction，再汇总24个样本。O1/O3参考O0；O7参考O5；O8参考O6。O5/O6本身是各组参考，不用“自比MSE=0”宣称源格式量化无误差。

| 后端 | 输出参考 | 512³ Median MSE | 512³ Mean MSE | 1024³ Median MSE | 1024³ Mean MSE |
|---|---|---:|---:|---:|---:|
| O1 | O0 | 1.203339e-9 | 1.125533e-9 | 2.245970e-9 | 2.011828e-9 |
| O3 | O0 | 8.518307e-4 | 1.005515e-3 | 1.471464e-3 | 1.664079e-3 |
| O7 | O5 | 6.864844e-4 | 6.688877e-4 | 1.191782e-3 | 1.082882e-3 |
| O8 | O6 | 5.594329e-4 | 5.665864e-4 | 9.247019e-4 | 8.906055e-4 |

O3的误差还包含G128与O0的G32权重量化差异及Q4转换误差，不应全部归因于求和顺序。不同参考组的MSE不能直接作为统一精度排名。计时边界调整前后，288项逐样本MSE完全一致。

## 性能差异

下面是24样本配对速度比的中位数，计算为“基线延迟÷该后端延迟”；大于1表示更快，不是吞吐提升百分比。

| 比较 | 512³ GEMM | 512³ Cold | 1024³ GEMM | 1024³ Cold |
|---|---:|---:|---:|---:|
| O3 / O1 | 1.500× | 1.000× | 2.056× | 1.484× |
| O7 / O5 | 1.000× | 0.769× | 1.211× | 1.000× |
| O8 / O6 | 1.000× | 0.800× | 1.211× | 1.054× |

512³ 时，O3/O7/O8 仅启动32个CTA，不足以覆盖108个SM；1024³ 为128个CTA，覆盖有所改善。kernel启动、pipeline初始化、metadata准备和在线转换在小矩阵中占比提高，因此延迟不会按矩阵运算量简单缩放。

1024³ 时，O7/O8 用较低的GEMM时间换来了更高的激活转换开销：A转换约10.6/10.2µs，而O5/O6约6.3/6.9µs。结果是O7的Cold中位数与O5相同；O8的Cold仅显示约1.05×的小幅收益。512³ 时，两者Cold中位数分别比FP16基线高约30%和25%。这些结论只针对当前最佳大矩阵算法的尺寸实例，不代表专门调优的小矩阵INT4上限。

## 正确性与计时可靠性

正式模式的288项样本输出参考检查通过；3,456条记录输出均与同一实现的参考调用逐位一致。44项额外检查覆盖随机、全零、正负交替、宽scale回退、非默认stream及四模式一致性；非法源编码被拒绝。最终扩展的 Compute Sanitizer memcheck 为 **0 errors**。

同一正式entry的PTX/SASS确认两路原生INT4与异步搬运，未匹配probe，也未退化为INT8。O3含少量local load/store指令，不能宣称零spill；O7/O8正式entry未出现local load/store。原4096的O3/O7/O8 SASS与已验收版本一致，原O1的17个swizzled实例指令也保持一致。另有48项K=4096、小M/N原路径正确性回归通过；本轮没有重测4096³整体性能，也未改动SM120后端。

小尺寸计时仍有波动，**不能宣称全部CV低于3%**：主实验725/3456条记录的主阶段CV≥3%，其中512³为679条、1024³为46条。GEMM具体如下：

| 后端 | 512³ GEMM CV ≥ 3% | 1024³ GEMM CV ≥ 3% |
|---|---:|---:|
| O1 | 72/72 | 0/72 |
| O3 | 72/72 | 0/72 |
| O5 | 72/72 | 12/72 |
| O6 | 72/72 | 13/72 |
| O7 | 26/72 | 0/72 |
| O8 | 32/72 | 0/72 |

例如512³ O3的一轮200次计时，116次为10.240µs、84次为11.264µs，CV约4.74%，没有长尾。这里的1.024µs台阶是本次观测，不是对CUDA Event通用分辨率的断言。其他记录也存在真实长尾或提交间隙；Event时间不等于纯MMA执行时间。

对“主阶段CV≥3%且有观测超过中位数20%”的 **271项** 另做一次同口径复测。复测/原三轮中位数的比值中位数为1.00，但仍有139项CV≥3%；不覆盖主表、不择优拼接。约1µs或几个百分点的细微差异不宜作强结论，512³ O7/O8应视为没有明确GEMM优势。

本轮仅对512/1024正式O1计时包装移除空阶段Event，使Compute-only total与GEMM一致；GPU算法、指令及输出不变。初次未统一Event边界的诊断数据不纳入以上主表，也不作为优化收益。

## 结果文件与复现

本地与A100项目内均保留：

- [主实验汇总](../runs/a100_best_sizes_aligned_four24/analysis.json)、[原始计时](../runs/a100_best_sizes_aligned_four24/results.jsonl)、[输出正确性](../runs/a100_best_sizes_aligned_four24/correctness.jsonl)。
- [环境与扩展指纹](../runs/a100_best_sizes_aligned_four24/environment.json)、[数据来源](../runs/a100_best_sizes_aligned_four24/data_provenance.json)、[GPU状态](../runs/a100_best_sizes_aligned_four24/gpu_snapshots.jsonl)。
- [正式指令审计](../reports/a100_best_sizes/aligned_audit/audit.json)、[O1指令未变检查](../reports/a100_best_sizes/o1_timing_codegen_regression.json)、[内存检查日志](../reports/a100_best_sizes/aligned_memcheck.log)。
- [一次复测汇总](../runs/a100_best_sizes_aligned_noise_recheck/summary.json)。

执行实验的源码commit为 `738edc2870d0acca656ef6ad8d005b1a519d16e5`。扩展SHA-256为 `41c681091b2986a3aea915429ff68e5751cfab13bfa70d5a892b4609d1085b8f`；主结果SHA-256为 `5b31dd3b1e7a5560e101b56d3dc4b64adf432e1a886821ec4f6b8123e60dee82`。

Git中另保存[原始计时与验证数据压缩包](../reports/a100_best_sizes/raw_results.tar.gz)。在新克隆、尚无这些结果目录的项目中，执行 `tar -xzf reports/a100_best_sizes/raw_results.tar.gz --keep-old-files` 可恢复上述JSON/JSONL及验证日志；完整PTX/SASS另外保留在本地和A100的 `reports/a100_best_sizes/aligned_audit/`。

在A100已有环境中复现，所有输出目录须未存在：

```bash
cd /home/zlouyang/ADAngel_oyzl
conda activate adangel-a100
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX="$CUDA_HOME/bin/nvcc" CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11

ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=2 \
  python setup.py build_ext --build-temp build/sm80-production --inplace

python scripts/benchmark_a100_best_sizes.py --validate-only \
  --output runs/a100_sizes_reproduce_validation

python scripts/audit_sm80_best_sizes.py \
  --output reports/a100_sizes_reproduce_audit

python scripts/benchmark_a100_best_sizes.py \
  --sizes 512 1024 --samples 24 --rounds 3 \
  --warmup 1000 --repeats 200 --inner 100 \
  --output runs/a100_sizes_reproduce

python scripts/analyze_a100_best_sizes.py \
  --input runs/a100_sizes_reproduce \
  --output runs/a100_sizes_reproduce/analysis.json
```

审计中的4096二进制回归需要项目中已有的v89/v99 codegen证据；正常正式kernel执行不依赖这些旧候选文件。无需重新采集4096 trace，也不需要安装或配置新的环境。


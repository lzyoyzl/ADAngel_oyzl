# ADAngel SM120 MXFP4 / 任意比特实验

本仓库实现 RTX 5090（SM120）上的 O0/O1/O2/O3/O4 对照实验。固定计算
`Y = A @ W.T`，`M=N=K=4096`，最终累加与输出均为 FP32。实验只统计：

1. 量化/反量化转换开销；
2. GEMM-only 与 cold/steady-state 端到端性能；
3. O1、O2、O3、O4 相对 O0 输出的 MSE。

显存占用、Roofline 和完整 GPU Profiling 不属于本实验。仓库只提供一次性的
SASS/PTX 指令审计，防止某个实现静默退化为 CUDA Core 或软件模拟。

> **正式结果的硬约束**：必须在 RTX 5090 上用 `sm_120a` 原生扩展生成。
> Python 参考后端用于正确性测试和开发，不允许写入正式性能结果。

## 当前实现状态

数据采集/准备、五配置语义参考、正式调度、统计、MSE、四表四图和防误跑能力门
已经实现。O0 正式后端使用预分配 CUDA kernel 完成 MXFP4→FP16 权重
反量化和 INT8→FP16 激活反量化，再由 cuBLASLt 执行行主序 FP16×FP16、FP32
累加/输出的 GEMM。算法搜索在计时区间外完成，并只接受同时声明 HMMA、FP16 输入和
FP32 累加的候选；所选 algorithm ID、数值实现 flags 与 workspace 大小会写入结果。

O1 正式后端也已实现：E2M1 nibble 由 CUDA kernel 精确转换为 `2*E2M1` INT8 基值。
production 现指向 `register_128x64_k64_scale_shared_row_dedup`：单次 TMA
cooperative warp-specialized kernel 中，每个 `128x64x64` CTA 由 1 个 producer warp
和 16 个 consumer warp 协作。三阶段 pipeline 每个 stage 搬运两个相邻 K32 group，
但两次 CuTe `m16n8k32` signed-INT8 MMA、scale 应用与 FP32 FMA 仍按 K32 独立且保持
group 0→127 的数值顺序。producer 对每个 CTA/column/group 只解码一次 W scale 并放入
stage-local shared memory；每线程只加载其 fragment 所需的两个唯一 row A scale。
INT32 partial 始终留在寄存器，最终每个输出元素只写回一次。

共享内存仍用于三阶段 TMA A/W pipeline 和 CTA 内 W scale 分发，不再用于 partial
中转。保留的 `shared_partial` 与 `register_64x32` 分别作为历史 baseline 和上一版
production。`register_128x128` 使用
`128x128x32` CTA、1 producer/16 consumer，只作为 CTA 消融；RTX 5090 资源审计发现
该候选存在 local-memory spill，因此不得晋升。O2 继续使用 `128x128x256`；不强制
O1/O2 完整 CTA 相同。

O2 正式后端现已实现。融合 CUDA kernel 将 `A_int8*A_scale` 按 K32 计算 amax、生成
UE8M0 scale、执行 E2M1 RNE 量化并完成 nibble packing；随后把自然顺序的 A/W scale
分别重排为 CUTLASS SFA/SFB physical layout。GEMM 使用 CUTLASS 4.5.2
`OpClassBlockScaledTensorOp`、`128x128x256` CTA tile、TMA 与
`KernelTmaWarpSpecializedCooperative`，执行原生 MXFP4 block-scaled MMA、FP32
累加和输出。W 的 packed MXFP4 数值不复制、不转置。

O3/O4 使用独立的 G128 权重副本，并按论文的 Split/Bitwise 算术实现。O3 将 A8
按 two's-complement 原始位拆成低 U4 和高 S4，满足
`A8=A_low_u4+16*A_high_s4`；G128 MXFP4 权重用 RNE 映射到 Q4。正式 kernel 采用
`128x16x128` CTA、两阶段 TMA、1 producer/16 consumer、两类 `m16n8k64`
INT4 MMA；A/B subbyte fragment 通过显式 `LDSM` 从 TMA staging shared memory
装入寄存器，在寄存器中重构两个 partial、应用 G128 scale，并只写一次 FP32 输出。

O4 将 A8 拆成系数 `[1,2,4,8,16,32,64,-128]` 的 8 个 bitplane，将 Q4 权重拆成
系数 `[1,2,4,-8]` 的 4 个 bitplane；每个 G128 执行 8×4=32 个
`m16n8k128.b1.and.popc` BMMA 并在寄存器中重构。正式实现
`m64_n64_k512_optimized` 采用 `64x64x512` CTA；每个两阶段 TMA pipeline stage
搬运四个连续但仍独立缩放的 G128。16 个 consumer warp 排列为 `4x4`，每 warp
覆盖两个相邻 N8 fragment；每个 G128 的四个 W bitplane fragment 只加载一次，
并用两条独立 INT32 reconstruction chain 缩短 BMMA 后处理依赖。为保留转换开销实验口径，当前 bitplane
生成作为独立 GPU conversion 计时；它对齐论文 Bitwise 算术，但不宣称实现论文的
selective fusion。

O0/O1/O2/O3/O4 capability 均已在源码中启用，但任何新源码仍必须在目标 RTX 5090 上重新
编译，并通过下述数值、计时与同 kernel 指令审计后才能生成正式结果。验收成功时
`python -m adangel doctor --require-native` 应报告 `available: true`；这个门不能用
reference 或 ISA probe 绕过。

## 目录说明

```text
ADAngel_oyzl/
├── configs/
│   ├── experiment/       # 尺寸、预热/重复次数、计时与统计口径
│   ├── machine/          # RTX 5090、sm_120a、CUDA/CUTLASS 约束
│   └── trace/            # Llama-2-7B 层与 projection 采样清单
├── csrc/
│   ├── common/           # CUDA 公共校验与辅助代码
│   └── sm120/            # O0–O4 转换、GEMM、绑定与 PTX microkernel
├── environment/          # Miniconda 环境定义；不包含驱动或系统 CUDA
├── include/adangel/      # C++/CUDA 公共数据结构和原生接口声明
├── python/adangel/
│   ├── analysis/         # 聚合 JSONL，生成四张主表和四张主图
│   ├── benchmark/        # CUDA Event 计时、MSE、环境与实验调度
│   ├── ops/              # 原生扩展加载、能力检查和统一 dispatch
│   ├── quantization/     # INT8、MXFP4、Q4、Split 和 bitplane 参考实现
│   ├── reference/        # O0–O4 可读的 FP32 语义参考实现
│   └── trace/            # trace 采集、样本 schema 和磁盘格式
├── scripts/              # 环境激活、检查、采集、准备、运行、汇总与指令审计入口
├── tests/
│   ├── unit/             # 16 种 E2M1 编码、UE8M0、mapping、指标测试
│   └── integration/      # 小矩阵 O0–O4 与参考实现交叉验证
├── third_party/cutlass/  # CUTLASS 固定版本/commit 元数据
├── third_party/cutlass-src/ # fetch 脚本生成的源码目录（不提交）
├── data/                 # 本地 trace/量化样本；默认不提交大文件
├── runs/                 # 每次运行的 config/environment/results
├── reports/              # 最终表格与图片
└── docs/                 # 数据格式、实验协议和 microscale layout 说明
```

## 实验语义

公共输入均已提前准备好，公共准备时间不计入 O0–O4：

```text
A_int8    [4096, 4096] int8     A_scale [4096]      fp32
W_mxfp4   [4096, 2048] uint8    W_scale [4096,128]  UE8M0 uint8
W_mxfp4_g128 [4096,2048] uint8  W_scale_g128 [4096,32] UE8M0 uint8
W_q4      [4096, 2048] uint8
Y         [4096, 4096] fp32
```

- **O0**：W 从 MXFP4 反量化为 FP16，A 从 INT8 反量化为 FP16，然后执行
  FP16 Tensor Core GEMM，FP32 累加/输出。
- **O1**：W 的 E2M1 编码精确映射为 `2*E2M1` 的 INT8 基值；每个 K32
  块执行 INT8 MMA，INT32 partial 转 FP32 后乘 `A_scale*W_scale/2` 并累加。
- **O2**：A 先反缩放再按 K32 重量化为 MXFP4；W 保持 MXFP4，执行
  SM120 block-scaled MXFP4 MMA，FP32 累加/输出。
- **O3**：A8 拆为 U4 low/S4 high，G128 MXFP4→Q4；每组用两条 INT4 MMA 路径
  重构 `P_low+16*P_high`，再乘 `A_scale*W_scale_g128`。
- **O4**：A8/Q4 分别拆为 8/4 个 two's-complement bitplane；每个 G128 用
  32 个 AND-POPC BMMA 重构整数点积，再乘相同的 G128 scale。

详细定义见 [实验协议](docs/experiment_protocol.md)、[数据格式](docs/data_format.md)、
[O0–O4 最终实现与正式实验结果](docs/o0_o4_final_results_report.md)、
[O1–O4 Nsight Compute profiling](docs/ncu_profiling.md)、
[O3/O4 NCU 性能瓶颈分析](docs/o3_o4_ncu_bottleneck_report.md)、
[O0/O1/O2 后端与测量报告](docs/o0_o1_o2_backend_report.md)、
[O3/O4 Split/Bitwise 后端报告](docs/o3_o4_backend_report.md)、
[Miniconda 配置指南](docs/miniconda_setup.md)、[本机验证记录](docs/local_validation.md)
和 [SM120 microscale layout](docs/mxfp4_scale_layout.md)。

## 服务器环境（Miniconda，不自动配置）

当前目标服务器基线已更新为 Ubuntu 24.04 x86_64、RTX 5090 32607 MiB、driver
595.58.03、`/usr/local/cuda-12.8`、nvcc 12.8.93、GCC/G++ 11.5.0 和 Git 2.43.0。
`nvidia-smi` 显示的 CUDA 13.2 是驱动支持上限，不是项目编译 Toolkit；本项目继续固定
使用 CUDA 12.8，不安装 CUDA 13.2。

Python 3.10.12 使用 Miniconda 环境 `adangel-sm120` 管理；PyTorch 固定为 2.7.1 cu128，
CUTLASS 固定为 v4.5.2 commit `db1c288993354c88e551c40c19a8fb93a774a241`。完整的
Miniconda 安装、依赖安装顺序、环境变量、构建、验收和故障排查见
[Miniconda 配置指南](docs/miniconda_setup.md)；精简依赖矩阵见
[服务器依赖清单](docs/server_dependencies.md)。机器可读版本位于 `environment/`、
`requirements/` 和 `configs/machine/rtx5090.yaml`。本仓库不会自动配置服务器，也不会
自动下载模型。

最短配置与构建顺序如下；首次安装 Miniconda 的步骤请直接按详细指南执行：

```bash
conda env create -f environment/miniconda-rtx5090.yml
conda activate adangel-sm120
source scripts/activate_server_env.sh
python -m pip install --upgrade pip==25.0.1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install PyYAML==6.0.2 typing_extensions==4.12.2
python -m pip install -r requirements/server-analysis.txt -r requirements/server-dev.txt
python -m pip check
bash scripts/fetch_cutlass.sh
python scripts/check_server_prereqs.py
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation --no-deps
python scripts/check_server_prereqs.py
python -m adangel doctor
```

`check_server_prereqs.py` 只读取版本，不执行安装或下载。原生 adapter 完成后，
`doctor --require-native` 会检查 RTX 5090、compute capability 12.0、锁定的
PyTorch/CUDA 构建、扩展编译目标及 kernel 能力。任何一项不满足都会退出，正式 benchmark
不会退回 reference。

## 1. 在模型服务器采集并传输 trace

Llama-2-7B 只需放在模型服务器；RTX 5090 服务器不加载或下载模型。模型服务器复用已有
`tensor_hook_cu128` 环境的方式是克隆出隔离环境，然后仅以 Python editable 模式安装本项目：

```bash
conda create --name adangel-trace --clone tensor_hook_cu128
conda activate adangel-trace
cd /path/to/ADAngel_oyzl
python -m pip install -e . --no-deps
```

不要在模型服务器运行 `ADANGEL_BUILD_CUDA=1`、获取 CUTLASS 或重新执行指令审计。确认
`transformers/datasets/accelerate/safetensors/sentencepiece/protobuf/PyYAML` 均可导入；只有缺包时才
按 `requirements/server-trace-optional.txt` 补装。正式采集和本地验证命令为：

```bash
python scripts/collect_trace.py \
  --model /absolute/path/to/Llama-2-7b-hf \
  --output data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml \
  --device cuda:0

python scripts/validate_raw_trace.py \
  --input data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml
```

采集器固定 WikiText-2 revision、随机种子和 4096-token FP16 prefill，生成 6 层 × 4 projection
共 24 个自包含 `.pt` 文件和 `trace_manifest.json`。随后将整个目录传至 RTX 5090：

```bash
rsync -avP --partial \
  data/raw/llama2_7b_prefill/ \
  zlouyang@<5090服务器地址>:/home/zlouyang/oyzl/ADAngel_oyzl/data/raw/llama2_7b_prefill/
```

在 RTX 5090 服务器再次运行相同的 `validate_raw_trace.py`；文件缺失、多余或 SHA-256
变化时会直接失败。完整环境核对、manifest 字段和故障处理见
[双服务器 trace 采集指南](docs/trace_collection.md)，字段定义见
[数据格式](docs/data_format.md)。

## 2. 生成公共输入

原 O0/O1/O2 prepared 目录保持不变。O3/O4 需要重新从同一份原始 FP16 trace
生成包含 K32 与 G128 两套权重表示的 v3 数据，建议使用新目录，避免覆盖既有结果：

```bash
python scripts/prepare_trace.py \
  --input data/raw/llama2_7b_prefill \
  --output data/prepared/llama2_7b_prefill_o0_o4 \
  --config configs/experiment/o0_o1_o2_o3_o4_4096.yaml \
  --trace-config configs/trace/llama2_7b_prefill.yaml
```

该步骤执行 FP16→逐行 INT8、FP16→K32 MXFP4、FP16→G128 MXFP4 和
G128 MXFP4→Q4，只执行一次。它输出
`A_int8/A_scale/W_mxfp4/W_scale/W_mxfp4_g128/W_scale_g128/W_q4` 以及样本
manifest；其耗时不会进入任何 variant。O3/O4 在线转换仍从公共 `A_int8` 和
G128 MXFP4 权重开始，并按实验模式单独计时。

## 3. 正确性测试

CPU 上可先运行不依赖 GPU 的编码测试：

```bash
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

修改任一原生后端后，先在 RTX 5090 上重新构建，再运行不需要模型或 trace 的专项验收：

```bash
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation --no-deps
python -c "import torch; import adangel._sm120 as m; print(dict(m.capabilities()))"
python scripts/validate_o0.py
python -m pytest tests/integration/test_sm120_o0.py -q --run-sm120
python scripts/validate_o1.py
python -m pytest tests/integration/test_sm120_o1.py -q --run-sm120
python scripts/validate_o2.py
python -m pytest tests/integration/test_sm120_o2.py -q --run-sm120
python scripts/validate_o3.py
python scripts/validate_o4.py
python -m pytest tests/integration/test_sm120_o3_o4.py -q --run-sm120
```

O3/O4 内部候选只用于配对消融，不改变公开 `run_o3/run_o4` 接口。可先用
`--samples 1 --warmup 5 --repeats 20` 做 smoke，再按 24 样本正式口径执行：

```bash
python scripts/benchmark_o3_o4_implementations.py \
  --variant o3 --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output reports/optimization/o3_candidates_24samples.json

python scripts/benchmark_o3_o4_implementations.py \
  --variant o4 --data data/prepared/llama2_7b_prefill_o0_o4 \
  --implementations m64_n64_k512_optimized \
  --output reports/optimization/o4_winner_24samples.json
```

正式提升要求：候选输出/MSE 回归通过，compute-only 配对加速 bootstrap 95% CI
下界大于 1，同一 PTX/SASS entry 包含 TMA 与目标 MMA，并且无 `LDL/STL`、
`STACK=0, LOCAL=0`。未满足者保留为内部消融，不得写入 production 结果。

预期 `o0_fp16_tc`、`o1_int8_tc`、`o2_mxf4_block_scale` 和
`o2_cutlass_tiled`、`o3_int4_tc`、`o3_tma_warp_specialized`、`o4_int1_tc` 和
`o4_tma_warp_specialized` 均为 `true`，五个验证脚本必须输出 `"passed": true`。
O0 元数据必须报告 HMMA、`CUBLAS_COMPUTE_32F` 和 FP32 输出；
O1 production 必须报告
`implementation_key=register_128x64_k64_scale_shared_row_dedup`、
`kernel_symbol=adangel_o1_register_partial_128x64_k64_scale_shared_row_dedup`、
`mma_api=cute::MMA_Atom`、`mma_shape=m16n8k32`、
`partial_storage=register`、`shared_partial_redistribution=false`、
`cta_tile=[128,64,64]`、`groups_per_pipeline_stage=2`、
`row_scale_loads_per_thread=2`、`global_partial_buffer=false` 和
`output_stores_per_element=1`。
O1 验证会逐元素核对 MXFP4→INT8 映射，并用 `rtol=1e-3, atol=1e-3` 对照可扩展
语义参考。O2 验证会逐元素核对激活 E2M1 编码、RNE、packing 和 UE8M0 scale，
执行 SFA/SFB layout probe，并检查 CUTLASS TMA cooperative warp-specialized 元数据。
O3/O4 还会逐元素核对 Q4/Split/bitplane 转换，并分别对照 G128 INT4 和
8×4 BMMA 语义参考。五个脚本都会验证四种计时模式的阶段集合。全部通过后，
`python -m adangel doctor --require-native` 应报告整体可用。

小矩阵通过后再执行 O1/O2 正式形状验收：

```bash
python scripts/validate_o1.py --implementation shared_partial \
  --m 4096 --n 4096 --k 4096 --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o1_4096_shared_baseline.json
python scripts/validate_o1.py --implementation register_64x32 \
  --m 4096 --n 4096 --k 4096 --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o1_4096_register_64x32.json
python scripts/validate_o1.py \
  --implementation register_128x64_k64_scale_shared_row_dedup \
  --m 4096 --n 4096 --k 4096 --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o1_4096_register_128x64_k64_production.json
python scripts/validate_o1.py --implementation register_128x128 \
  --m 4096 --n 4096 --k 4096 --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o1_4096_register_128x128.json
```

24 样本配对 A/B 与 CTA 消融：

```bash
python scripts/benchmark_o1_implementations.py \
  --data data/prepared/llama2_7b_prefill \
  --output reports/o1_register_partial_ab.json \
  --warmup 50 --repeats 200 --conversion-inner-repeats 100 \
  --cv-policy diagnostic \
  --max-paired-retries 5
```

默认 `--cv-policy diagnostic` 面向未锁频的 GeForce：所有 CV、离群值和定点重试均
完整保留，但少量动态 Boost/调度离群不单独阻断晋升。性能判据使用同进程、同输入、
交替顺序的24样本配对中位数；要求 correctness/MSE 通过、compute-only speedup 的
bootstrap 95% CI 下界大于1，并通过无spill审计。需要锁频环境的严格复现实验时使用
`--cv-policy strict`，此时所有计时阶段还必须 `CV<3%`，未解决重试会令脚本失败。
两种策略都禁止删除单边慢样本或按延迟挑选重试结果。旧 shared-partial 和
`register_64x32` run 不能冒充当前 `128x64x64` production 结果。

若正式runner含偶发CV失败，使用成组定向重测；任一`(sample, mode)`失败时必须同时
重跑O0/O1/O2，并只接受三者全部稳定的第一次尝试：

```bash
python scripts/retry_unstable_run.py \
  --run runs/rtx5090_o1_128x64_production \
  --data data/prepared/llama2_7b_prefill \
  --max-attempts 5
```

脚本将原始结果备份为`results.initial.jsonl`，把所有尝试写入
`retry_attempts.jsonl`，把替换策略和SHA-256写入`retry_audit.json`。它不按延迟挑选
结果，也不会只重测某个variant；存在未解决target时不会替换正式`results.jsonl`。
正式runner与重测脚本会在初始化CUDA前通过`nvidia-smi`记录compute process，但默认
允许共享GPU继续运行。正式run额外写出`timing_preflight.json`；若启动时已有并发任务，
该文件会明确警告它们可能增加方差和离群。需要独占GPU的严格复现时才显式添加
`--require-idle-gpu`，此时检测到compute process会立即拒绝运行。
若外部任务在重测中途启动，可安全中止后用同一命令继续；脚本会从
`retry_attempts.jsonl`恢复已经通过的第一次全稳定attempt，只补测未完成target。

```bash
python scripts/validate_o2.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o2_4096_validation.json

python scripts/validate_o3.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o3_4096_validation.json

python scripts/validate_o4.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 50 --repeats 200 --max-cv-percent 3.0 \
  | tee reports/o4_4096_validation.json
```


三组专项测试通过后，再执行全部原生集成测试和 PTX layout microkernel：

```bash
python -m pytest tests/integration -q --run-sm120
python -m adangel verify-layout --require-native
```

覆盖全零、最大有限值、正负交替、随机值和饱和值。microkernel 对 FP64 软件
参考的容差为 `rtol=1e-3, atol=1e-3`，输出不得含 NaN/Inf。

## 4. 一次性指令审计

```bash
EXTENSION_DIR=$(python -c "import torch, pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit
```

审计应找到：O0 的 FP16 Tensor Core 指令、O1 的 INT8 Tensor Core 指令、
O2 的 `kind::mxf4 ... ue8m0` block-scaled MMA、O3 的 U4×S4/S4×S4 INT4 MMA，
以及 O4 的 B1 AND-POPC BMMA。脚本要求 O1 的 TMA 与 signed INT8
MMA 位于选定的 production entry；保留的寄存器候选也必须在各自同一 entry 内包含
TMA+IMMA。O1 production `register_128x64_k64_scale_shared_row_dedup` 必须满足
SASS 无 `LDL/STL` 且 resource usage 为 `STACK=0, LOCAL=0`。`register_128x128` 是
可选 CTA 消融；若出现 spill，summary 将其标记为 `DISQUALIFIED`，但不会阻断无 spill
的 production；若把有
spill的128×128实现选为production，审计仍会失败。O2 的
TMA 与 MXFP4 block-scaled MMA 位于同一个正式 CUTLASS PTX/SASS entry，并明确排除
`o2_mxf4_layout_probe`。summary 记录 production/candidate symbol、spill 检查与
`status=PASS`，资源使用写入 `extension.resources.txt`。O3/O4 审计会按模板配置精确
定位 production entry，并逐一列出内部候选的寄存器/stack/local 状态；正式 O3/O4
要求同一 entry 内同时出现 TMA 和目标 MMA，且无 `LDL/STL`、
`STACK=0, LOCAL=0`。有 spill 的候选标记为 `DISQUALIFIED`，不影响已通过的
production。该审计不是完整 profiling。

## 5. 正式运行

单次 smoke（先确认流程和温度控制）：

```bash
python -m adangel run \
  --config configs/experiment/o0_o1_o2_o3_o4_4096.yaml \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/smoke_o0_o4 \
  --samples 1 --warmup 5 --repeats 10 --require-native
```

正式 24 样本实验：

```bash
python -m adangel run \
  --config configs/experiment/o0_o1_o2_o3_o4_4096.yaml \
  --data data/prepared/llama2_7b_prefill_o0_o4 \
  --output runs/rtx5090_o0_o4_k512 \
  --require-native
```

当前 RTX 5090 的 `runs/rtx5090_o0_o4_k512` 24 样本 GEMM-only median 为：
O1 `0.622136 ms`、O3 `2.223352 ms`、O4 `7.632488 ms`。最新 O4 候选相对上一版
`128x64x256` production 的逐样本配对几何平均加速为 `1.0806x`，bootstrap 95% CI
为 `[1.0789x, 1.0827x]`。这里的 O3 `2.223352 ms` 是显式 LDSM 晋升前的历史结果；
新 production 保持 `128x16x128`，仅把标量 shared fragment load 改为显式 LDSM，
当前提交已生成 `runs/rtx5090_o0_o4_o3_ldsm`，但共享 GPU 外部进程导致451/480条
记录 CV 超过3%，因此该 run 只用于确认 metadata/MSE，不替换上面的稳定绝对性能表。
详细消融、资源审计、性能边界与 MSE 见
`docs/o3_o4_backend_report.md`。

运行顺序按样本交错 O0/O1/O2/O3/O4；使用单 stream、CUDA Event、预热 50 次、测量
200 次。计时Event必须在预热前完成创建，预热同步与第一条正式测量之间不得创建
Event、分配显存或做文件 I/O，避免GPU降频后在测量区间重新升频。每个 run 生成：

- `config.yaml`：解析后的完整配置；
- `environment.json`：GPU、driver、CUDA、PyTorch、git commit 与扩展能力；
- `results.jsonl`：schema v2；每个样本/variant/mode 的原始统计、计时方法与 MSE。

正式配置启用 `CV < 3%` 门限。超过门限的记录标为失败，不能静默进入主表。
转换阶段采用批量摊销计时：O0-W、O0-A、O1-W、O2-W-layout 和 O2-A 均在独立
CUDA Event 区间内连续执行 `timing.conversion_inner_repeats=100` 次，再以总耗时
除以 100。`conversion_only/total` 同样对完整转换序列执行100次后摊销。这样可降低
CUDA Event 分辨率和少量调度离群值对微秒级转换kernel的相对影响。

端到端路径保持单次直接计时：`compute_only/total`、`cold/total` 和
`steady_state/total` 的每个原始样本都只执行一次对应路径；cold仍只包含一次权重转换，
steady-state仍缓存静态权重转换。批量组件测量与直接total相互隔离，所以各阶段median
之和不要求与direct total严格相等，端到端结论以direct total为准。每条
`results.jsonl` 记录的 `timing_method` 会保存实际策略、inner repeats和该mode的
total计时语义。
转换开销主表把该阶段记录为 o2/weight_conversion（图中即 O2-W-layout）。
转换吞吐按有效 tensor 的逻辑读写字节计算，不把未触碰的 physical-layout padding 计入：

- O2-W：2*N*(K/32)，即读取自然 W_scale 并写出有效 SFB scale。
- O2-A：M*K + 4*M + M*K/2 + 3*M*(K/32)。

## 6. 生成四表四图

```bash
python -m adangel analyze \
  --run runs/rtx5090_main \
  --output reports/rtx5090_main
```

默认分析使用 `--stability-policy strict`，任何 `CV>=3%` 的记录都会阻止制图。若明确
选择不锁频，且动态Boost/桌面图形调度造成大量CV标记，可显式使用：

```bash
python -m adangel analyze \
  --run runs/rtx5090_main \
  --output reports/rtx5090_main_dynamic \
  --stability-policy diagnostic \
  --diagnostic-note "Shared GPU run; concurrent workloads may explain isolated CV outliers."
```

diagnostic不会删除或重写原始样本；四张主表保留`valid/stable_cv`列，额外生成
`tables/00_stability.csv`和`report_metadata.json`，所有图片标题标明动态Boost诊断口径。
`--diagnostic-note`会把已知的共享GPU、动态Boost或调度干扰原因写入metadata。该模式
用于报告中位数与MSE，P5/P95和CV仍必须一并披露，不能称为独占GPU或全部CV通过的
稳定性结果；不得删除离群样本或只保留更快的重测结果。

输出：

```text
reports/rtx5090_main/tables/
  01_conversion_overhead.csv
  02_gemm_only.csv
  03_end_to_end.csv
  04_mse.csv
reports/rtx5090_main/figures/
  01_conversion_overhead.png
  02_gemm_only.png
  03_end_to_end_breakdown.png
  04_mse_boxplot.png
```

MSE 以每个样本保存的一次 O0 FP32 输出为唯一主参考，O1/O2/O3/O4 转 FP64 后做
reduction；报告逐样本值以及 24 样本的 median、IQR、最大值和 bootstrap 95% CI。

## 常用诊断

```bash
python -m adangel doctor
python -m adangel show-config --config configs/experiment/o0_o1_o2_o3_o4_4096.yaml
python -m adangel run --help
```

如果机器不是 RTX 5090、原生扩展未编译、CUTLASS commit 不匹配或任一正式 kernel
未报告 block-scaled MMA 能力，`--require-native` 会停止运行。小矩阵开发验证可在
Python API 的 `run_o0/run_o1/run_o2/run_o3/run_o4` 中显式传入 `backend="reference"`；reference
后端禁止 M=N=K=4096 的正式计时，也不会生成可汇总的正式性能结果。

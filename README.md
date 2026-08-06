# ADAngel SM120 MXFP4 实验

本仓库实现 RTX 5090（SM120）上的 O0/O1/O2 对照实验。固定计算
`Y = A @ W.T`，`M=N=K=4096`，最终累加与输出均为 FP32。实验只统计：

1. 量化/反量化转换开销；
2. GEMM-only 与 cold/steady-state 端到端性能；
3. O1、O2 相对 O0 输出的 MSE。

显存占用、Roofline 和完整 GPU Profiling 不属于本实验。仓库只提供一次性的
SASS/PTX 指令审计，防止某个实现静默退化为 CUDA Core 或软件模拟。

> **正式结果的硬约束**：必须在 RTX 5090 上用 `sm_120a` 原生扩展生成。
> Python 参考后端用于正确性测试和开发，不允许写入正式性能结果。

## 当前实现状态

数据采集/准备、三配置语义参考、正式调度、统计、MSE、四表四图和防误跑能力门
已经实现。`csrc/sm120` 中也提供三类目标 MMA 的 ISA probe。当前提交尚未在 RTX
5090 上编译验证，`o2_cutlass.cu` 的 publication-performance CUTLASS adapter 因而
保持关闭，`capabilities()` 会返回 false，正式 `run` 会按设计拒绝启动。完成该 adapter
并在 5090 上通过数值、layout 和指令审计后，才允许把能力位改为 true。这个门不能
用 reference 或单 warp probe 绕过。

## 目录说明

```text
ADAngel_oyzl/
├── configs/
│   ├── experiment/       # 尺寸、预热/重复次数、计时与统计口径
│   ├── machine/          # RTX 5090、sm_120a、CUDA/CUTLASS 约束
│   └── trace/            # Llama-2-7B 层与 projection 采样清单
├── csrc/
│   ├── common/           # CUDA 公共校验与辅助代码
│   └── sm120/            # O0/O1/O2 转换、GEMM、绑定与 PTX microkernel
├── include/adangel/      # C++/CUDA 公共数据结构和原生接口声明
├── python/adangel/
│   ├── analysis/         # 聚合 JSONL，生成四张主表和四张主图
│   ├── benchmark/        # CUDA Event 计时、MSE、环境与实验调度
│   ├── ops/              # 原生扩展加载、能力检查和统一 dispatch
│   ├── quantization/     # INT8、E2M1、UE8M0、K32 packing 参考实现
│   ├── reference/        # O0/O1/O2 可读的 FP32 语义参考实现
│   └── trace/            # trace 采集、样本 schema 和磁盘格式
├── scripts/              # 采集、准备、运行、汇总与指令审计入口
├── tests/
│   ├── unit/             # 16 种 E2M1 编码、UE8M0、mapping、指标测试
│   └── integration/      # 小矩阵 O0/O1/O2 与参考实现交叉验证
├── third_party/cutlass/  # CUTLASS 固定版本/commit 元数据
├── third_party/cutlass-src/ # fetch 脚本生成的源码目录（不提交）
├── data/                 # 本地 trace/量化样本；默认不提交大文件
├── runs/                 # 每次运行的 config/environment/results
├── reports/              # 最终表格与图片
└── docs/                 # 数据格式、实验协议和 microscale layout 说明
```

## 实验语义

公共输入均已提前准备好，公共准备时间不计入 O0/O1/O2：

```text
A_int8    [4096, 4096] int8     A_scale [4096]      fp32
W_mxfp4   [4096, 2048] uint8    W_scale [4096,128]  UE8M0 uint8
Y         [4096, 4096] fp32
```

- **O0**：W 从 MXFP4 反量化为 FP16，A 从 INT8 反量化为 FP16，然后执行
  FP16 Tensor Core GEMM，FP32 累加/输出。
- **O1**：W 的 E2M1 编码精确映射为 `2*E2M1` 的 INT8 基值；每个 K32
  块执行 INT8 MMA，INT32 partial 转 FP32 后乘 `A_scale*W_scale/2` 并累加。
- **O2**：A 先反缩放再按 K32 重量化为 MXFP4；W 保持 MXFP4，执行
  SM120 block-scaled MXFP4 MMA，FP32 累加/输出。

详细定义见 [实验协议](docs/experiment_protocol.md)、[数据格式](docs/data_format.md)、
[本机验证记录](docs/local_validation.md)
和 [SM120 microscale layout](docs/mxfp4_scale_layout.md)。

## 服务器依赖（这里只给清单，不自动配置）

正式环境固定为 Ubuntu 22.04 x86_64、Python 3.10.12、CUDA Toolkit 12.8 Update 1
（nvcc 12.8.90）、PyTorch 2.7.1 cu128、CUTLASS v4.5.2 commit
`db1c288993354c88e551c40c19a8fb93a774a241`、GCC/G++ 11、CMake 3.24+、Ninja 和
C++17。Linux NVIDIA driver 至少为 570.124.06；建议使用服务器已有的更新版生产驱动。

完整的系统包、Python 锁定依赖、PyTorch cu128 安装顺序、环境变量、磁盘需求和逐项验收命令见
[服务器依赖清单](docs/server_dependencies.md)。可机器读取的版本位于 `requirements/` 和
`configs/machine/rtx5090.yaml`。本仓库不会自动安装驱动/CUDA/Python 包，也不会自动下载模型。

服务器人工配置完成后，先执行只读检查，再获取固定 CUTLASS 源码并构建：

```bash
python scripts/check_server_prereqs.py --skip-cutlass
bash scripts/fetch_cutlass.sh
export ADANGEL_CUTLASS_ROOT="$PWD/third_party/cutlass-src"
export TORCH_CUDA_ARCH_LIST='12.0a'
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation
python scripts/check_server_prereqs.py
python -m adangel doctor --require-native
```

`check_server_prereqs.py` 只读取版本，不执行安装或下载。`doctor --require-native` 会检查 RTX
5090、compute capability 12.0、锁定的 PyTorch/CUDA 构建、扩展编译目标及 kernel 能力。
任何一项不满足都会退出，正式 benchmark 不会退回 reference。

## 1. 放置外部 trace（主流程，不下载模型）

本轮不在项目中下载或加载 Llama-2-7B。请把服务器外部已经采集好的 24 个 FP16 样本放到
`data/raw/llama2_7b_prefill/`。每个 `.pt` 文件必须包含
`sample_id/layer/projection/activation_fp16/weight_fp16`；层为 `0,6,12,18,24,31`，projection
为 `q_proj/k_proj/v_proj/o_proj`，A/W 均严格为 `[4096,4096]` FP16。完整 schema、命名和
校验规则见 [数据格式](docs/data_format.md)。

仓库保留 `scripts/collect_trace.py` 仅作为未来可选工具；当前主流程不调用它，也不需要安装
`requirements/server-trace-optional.txt`。

## 2. 生成公共输入

```bash
python scripts/prepare_trace.py \
  --input data/raw/llama2_7b_prefill \
  --output data/prepared/llama2_7b_prefill \
  --config configs/experiment/o0_o1_o2_4096.yaml \
  --trace-config configs/trace/llama2_7b_prefill.yaml
```

该步骤执行 FP16→逐行 INT8 和 FP16→K32 MXFP4，只执行一次。它输出
`A_int8/A_scale/W_mxfp4/W_scale` 以及样本 manifest；其耗时不会进入三种配置。

## 3. 正确性测试

CPU 上可先运行不依赖 GPU 的编码测试：

```bash
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

RTX 5090 上运行原生集成测试和 PTX layout microkernel：

```bash
python -m pytest tests/integration -q --run-sm120
python -m adangel verify-layout --require-native
```

覆盖全零、最大有限值、正负交替、随机值和饱和值。microkernel 对 FP64 软件
参考的容差为 `rtol=1e-3, atol=1e-3`，输出不得含 NaN/Inf。

## 4. 一次性指令审计

```bash
bash scripts/audit_instructions.sh build reports/audit
```

审计应找到：O0 的 FP16 Tensor Core 指令、O1 的 INT8 Tensor Core 指令，以及
O2 的 `kind::mxf4 ... ue8m0` block-scaled MMA。脚本只做存在性冒烟检查；它不是
完整 profiling。审计文件与实验环境一起归档。

## 5. 正式运行

单次 smoke（先确认流程和温度控制）：

```bash
python -m adangel run \
  --config configs/experiment/o0_o1_o2_4096.yaml \
  --data data/prepared/llama2_7b_prefill \
  --output runs/smoke \
  --samples 1 --warmup 5 --repeats 10 --require-native
```

正式 24 样本实验：

```bash
python -m adangel run \
  --config configs/experiment/o0_o1_o2_4096.yaml \
  --data data/prepared/llama2_7b_prefill \
  --output runs/rtx5090_main \
  --require-native
```

运行顺序按样本交错 O0/O1/O2；使用单 stream、CUDA Event、预热 50 次、测量
200 次，计时区间不分配显存、不做文件 I/O。每个 run 生成：

- `config.yaml`：解析后的完整配置；
- `environment.json`：GPU、driver、CUDA、PyTorch、git commit 与扩展能力；
- `results.jsonl`：每个样本/variant/mode 的原始统计与 MSE。

正式配置启用 `CV < 3%` 门限。超过门限的记录标为失败，不能静默进入主表。

## 6. 生成四表四图

```bash
python -m adangel analyze \
  --run runs/rtx5090_main \
  --output reports/rtx5090_main
```

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

MSE 以每个样本保存的一次 O0 FP32 输出为唯一主参考，O1/O2 转 FP64 后做
reduction；报告逐样本值以及 24 样本的 median、IQR、最大值和 bootstrap 95% CI。

## 常用诊断

```bash
python -m adangel doctor
python -m adangel show-config --config configs/experiment/o0_o1_o2_4096.yaml
python -m adangel run --help
```

如果机器不是 RTX 5090、原生扩展未编译、CUTLASS commit 不匹配或 O2 kernel
未报告 block-scaled MMA 能力，`--require-native` 会停止运行。小矩阵开发验证可在
Python API 的 `run_o0/run_o1/run_o2` 中显式传入 `backend="reference"`；reference
后端禁止 M=N=K=4096 的正式计时，也不会生成可汇总的正式性能结果。

# RTX 5090 服务器依赖与版本矩阵

本文只描述后续在服务器上需要的环境，不会在当前机器或服务器上自动安装任何组件，也不会
下载 Llama-2-7B。完整逐步命令见 [Miniconda 配置指南](miniconda_setup.md)，正式实验直接
接收外部采集的 24 个 trace 样本。

## 1. 固定平台与版本

| 组件 | 固定版本/要求 | 用途 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090，compute capability 12.0 | 唯一正式实验设备 |
| 显存 | 32607 MiB | 服务器 `nvidia-smi` 实测 |
| OS | Ubuntu 24.04 LTS x86_64 | 固定用户态基线 |
| NVIDIA driver | 595.58.03（最低仍为 570.124.06） | 保留服务器现有驱动 |
| `nvidia-smi` CUDA | 13.2 | 驱动支持上限，不是编译 Toolkit |
| CUDA Toolkit | 12.8，路径 `/usr/local/cuda-12.8` | 提供 SM120/PTX、headers、nvcc、cuBLASLt |
| nvcc | release 12.8，build 12.8.93 | 编译 `sm_120a` 扩展 |
| Python | 3.10.12，64-bit | 实验驱动 |
| 环境管理 | Miniconda，环境名 `adangel-sm120` | 隔离 Python 与构建工具 |
| PyTorch | 2.7.1，官方 cu128 wheel | CUDA tensor、扩展绑定、CUDA Event |
| CUTLASS | v4.5.2，commit `db1c288993354c88e551c40c19a8fb93a774a241` | 正式 SM120 tiled kernel |
| GCC/G++ | 11.5.0 | 服务器已有 CUDA host compiler |
| Git | 2.43.0 | 获取和校验 CUTLASS |
| CMake | 3.30.5 | Conda 环境内通过 pip 安装 |
| Ninja | 1.11.1.3 Python 包 | PyTorch 扩展并行构建 |
| C++ | C++17 | 统一编译标准 |

driver 595.58.03 比 CUDA 12.8 所需最低版本更新，能够通过向后兼容运行 CUDA 12.8
应用。不要因为 `nvidia-smi` 显示 13.2 而更换 CUDA Toolkit。不要使用 A100/H100 后端，
也不要把 `sm_120` 或 PTX JIT 替代 `sm_120a` 的正式构建。

## 2. 系统包

服务器已经提供且项目只读使用：

```text
NVIDIA driver 595.58.03
CUDA Toolkit /usr/local/cuda-12.8（nvcc 12.8.93）
gcc-11 / g++-11 11.5.0
Git 2.43.0
```

无需重新安装这些系统组件。Python、CMake、Ninja 和其余用户态包由 Miniconda 环境提供。
项目需要系统 Toolkit 编译器；只有 driver/runtime 或 PyTorch 自带 CUDA runtime 不足以编译
扩展。

## 3. Python 包

锁定文件按职责拆分：

- `requirements/server-core.txt`：运行与构建必需；
- `requirements/server-analysis.txt`：四表四图；
- `requirements/server-dev.txt`：测试和扩展构建工具；
- `requirements/server-trace-optional.txt`：仅未来自行采集模型 trace 时需要，本轮不安装。

PyTorch 必须从官方 cu128 wheel 索引安装。Miniconda 环境的完整手工顺序为：

```bash
conda env create -f environment/miniconda-rtx5090.yml
conda activate adangel-sm120
source scripts/activate_server_env.sh
python -m pip install --upgrade pip==25.0.1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/server-analysis.txt -r requirements/server-dev.txt
python -m pip install PyYAML==6.0.2 typing_extensions==4.12.2
python -m pip check
```

`server-core.txt` 是版本清单；其中 torch 的实际安装仍应使用上面的 cu128 专用索引，防止从默认
索引得到不符合要求的构建。模型 trace 已由外部提供时，不安装 transformers、safetensors 或
sentencepiece。

## 4. 构建变量

项目提供的激活脚本只设置当前 shell，避免污染其他 CUDA 项目：

```bash
conda activate adangel-sm120
source scripts/activate_server_env.sh
```

除原有变量外，脚本还固定 `CUDACXX=/usr/local/cuda-12.8/bin/nvcc`，并在外部没有预先
指定时使用 `CUDA_VISIBLE_DEVICES=0`。

CUTLASS 必须是可验证 commit 的 git checkout。仓库的 `scripts/fetch_cutlass.sh` 只在你主动执行时
获取固定 tag；项目导入、测试和 benchmark 都不会隐式联网。

构建命令（供服务器上手工执行）：

```bash
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation --no-deps
```

构建脚本会拒绝错误的 PyTorch/CUDA 构建或 CUTLASS commit。正式 runner 还会拒绝非 RTX 5090、
非 compute capability 12.0、能力位未完成或没有 `sm_120a` 的扩展。

## 5. 外部 trace 与磁盘

不需要模型文件。只需从外部复制以下数据：

```text
data/raw/llama2_7b_prefill/
  layer_00_q_proj.pt
  ...
  layer_31_o_proj.pt
```

共 24 个样本；每个文件包含 `[4096,4096]` FP16 的激活和权重。原始 trace 约 1.5 GiB，准备后
数据约 0.57 GiB；若保存 24 份 O0 FP32 输出，另需约 1.5 GiB。考虑构建缓存、审计文件、运行
副本和报告，建议项目所在盘至少预留 10 GiB。详细字段见 `docs/data_format.md`。

## 6. 只读验收

服务器配置完成后，以下命令只检查环境，不安装或下载内容：

```bash
python scripts/check_server_prereqs.py --skip-cutlass
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
nvcc --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
gcc-11 --version
g++-11 --version
git --version
cmake --version
ninja --version
git -C third_party/cutlass-src rev-parse HEAD
python scripts/check_server_prereqs.py
python -m adangel doctor
```

期望关键输出为：Ubuntu 24.04、RTX 5090、`(12, 0)`、torch `2.7.1+cu128`（显示形式可能
保留 local tag）、`torch.version.cuda == 12.8`、nvcc build `12.8.93`、GCC/G++ 11.5.0、
Git 2.43.0 和上表 CUTLASS 完整 SHA。检查脚本返回非零
表示依赖尚未满足，不应开始编译或实验。

## 7. 构建后的项目验收

```bash
python -m adangel doctor --require-native
python -m unittest discover -s tests/unit -p 'test_*.py' -v
python -m pytest tests/integration -q --run-sm120
python -m adangel verify-layout --require-native
EXTENSION_DIR=$(python -c "import torch, pathlib; import adangel._sm120 as m; print(pathlib.Path(m.__file__).parent)")
bash scripts/audit_instructions.sh "$EXTENSION_DIR" reports/audit
```

这些检查不进行模型下载。只有 `scripts/collect_trace.py` 被用户显式调用时才会访问模型仓库；本轮
主流程不调用该脚本。

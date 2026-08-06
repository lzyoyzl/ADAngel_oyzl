# RTX 5090 服务器 Miniconda 配置指南

本文用于在目标服务器上手工建立 ADAngel 实验环境。项目不会自动安装系统组件、修改
NVIDIA 驱动或下载大模型。命令默认在仓库根目录执行，Conda 环境名固定为
`adangel-sm120`。

## 1. 已确认的服务器基线

| 项目 | 服务器实际值 | 项目处理方式 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090 | 直接使用 GPU 0 |
| 显存 | 32607 MiB | 足够运行固定的 `4096^3` 实验 |
| Driver | 595.58.03 | 保留，不重装 |
| `nvidia-smi` CUDA | 13.2 | 代表驱动可支持的最高 CUDA 版本 |
| CUDA Toolkit | `/usr/local/cuda-12.8` | 编译时固定使用 |
| nvcc | 12.8.93 | 编译 `sm_120a` 扩展 |
| GCC/G++ | 11.5.0 | CUDA host compiler |
| Git | 2.43.0 | 获取并校验 CUTLASS |
| OS | Ubuntu 24.04 x86_64 | 由现有 Ubuntu 24.04 软件包版本确定；以 `/etc/os-release` 复核 |

`nvidia-smi` 中的 `CUDA Version: 13.2` 不是当前 shell 使用的 CUDA Toolkit 版本。
它表示驱动 595.58.03 能支持的最高 CUDA API 版本。项目实际使用
`/usr/local/cuda-12.8/bin/nvcc`，因此 `nvcc --version` 显示 12.8.93 完全正常。新驱动可以
运行用较旧 Toolkit 构建的应用，本项目不需要安装 CUDA 13.2。

先只读复核系统项：

```bash
cat /etc/os-release
uname -m
nvidia-smi
/usr/local/cuda-12.8/bin/nvcc --version
/usr/bin/gcc-11 --version
/usr/bin/g++-11 --version
git --version
```

预期分别包含 Ubuntu 24.04、`x86_64`、RTX 5090、driver 595.58.03、nvcc
`V12.8.93`、GCC/G++ 11.5.0 和 Git 2.43.0。系统项不匹配时先停止，不要在 Conda
环境中尝试替换 GPU 驱动或 `/usr/local/cuda-12.8`。

## 2. 安装 Miniconda

如果 `conda --version` 已可用，跳过本节。否则按 Miniconda 官方 Linux x86_64 方法以
普通用户安装，不需要 `sudo`：

```bash
cd /tmp
curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
```

把输出的 SHA-256 与
`https://repo.anaconda.com/miniconda/` 页面中同名安装包的官方值人工比较；一致后再执行：

```bash
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
conda config --set auto_activate_base false
conda --version
```

重新登录后若 `conda` 尚不可用，先执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

Miniconda 安装器本身会随时间更新；真正影响本实验的 Python 与 pip 依赖均在仓库内锁定。
配置完成后还会导出服务器实际环境快照。

## 3. 创建隔离环境

进入仓库根目录后用项目提供的环境文件创建环境：

```bash
conda env create -f environment/miniconda-rtx5090.yml
conda activate adangel-sm120
python --version
which python
```

预期 Python 为 `3.10.12`，路径位于
`$HOME/miniconda3/envs/adangel-sm120/bin/python`。不要使用系统 Python，也不要在
Conda `base` 环境中安装项目依赖。

## 4. 加载 CUDA 与编译器变量

每次打开新 shell，都先激活 Conda 环境并 source 项目脚本：

```bash
conda activate adangel-sm120
source scripts/activate_server_env.sh
```

脚本只设置当前 shell，不修改系统安装。它会设置：

```text
CUDA_HOME=/usr/local/cuda-12.8
CC=/usr/bin/gcc-11
CXX=/usr/bin/g++-11
CUDACXX=/usr/local/cuda-12.8/bin/nvcc
TORCH_CUDA_ARCH_LIST=12.0a
CUDA_VISIBLE_DEVICES=0（若外部未预先设置）
ADANGEL_CUTLASS_ROOT=<仓库>/third_party/cutlass-src
```

确认实际命令来自预期位置：

```bash
echo "$CUDA_HOME"
which nvcc
which gcc-11
which g++-11
echo "$CUDACXX"
echo "$TORCH_CUDA_ARCH_LIST"
```

## 5. 安装锁定的 Python 依赖

先升级到固定 pip，再从 PyTorch 官方 cu128 索引安装 torch。项目不需要 torchvision 或
torchaudio：

```bash
python -m pip install --upgrade pip==25.0.1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install PyYAML==6.0.2 typing_extensions==4.12.2
python -m pip install -r requirements/server-analysis.txt
python -m pip install -r requirements/server-dev.txt
python -m pip check
```

依赖职责如下：

| 文件 | 内容 |
|---|---|
| `requirements/server-core.txt` | torch、PyYAML、typing_extensions 的版本清单 |
| `requirements/server-analysis.txt` | NumPy、pandas、Matplotlib、Pillow |
| `requirements/server-dev.txt` | pytest、ruff、setuptools、wheel、Ninja、CMake |
| `requirements/server-trace-optional.txt` | 仅自行采集模型 trace 时使用；本实验不安装 |

不要直接从默认 PyPI 安装 `server-core.txt`，因为 torch 必须来自官方 cu128 索引。上述顺序
会让 torch 自动安装与其 wheel 匹配的 CUDA 用户态运行库；这些库不会替换系统驱动或
`nvcc`。

检查 Python 与 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

预期为 torch `2.7.1+cu128`、`torch.version.cuda == "12.8"`、RTX 5090 和
compute capability `(12, 0)`。

## 6. 获取固定 CUTLASS 并检查环境

项目固定 CUTLASS v4.5.2 commit
`db1c288993354c88e551c40c19a8fb93a774a241`：

```bash
bash scripts/fetch_cutlass.sh
git -C third_party/cutlass-src rev-parse HEAD
python scripts/check_server_prereqs.py
```

如果服务器暂时不能访问 GitHub，可在联网机器获取同一 commit 后完整复制
`third_party/cutlass-src/`，但必须保留 `.git` 目录，以便构建脚本验证 commit。只想在
CUTLASS 到位前检查其余项目时使用：

```bash
python scripts/check_server_prereqs.py --skip-cutlass
```

检查器是只读的，不会安装或下载任何内容。它会检查 Ubuntu、Python、环境变量、Python
包、RTX 5090、compute capability、driver、nvcc、GCC/G++、Git、CMake、Ninja 和
CUTLASS commit。

## 7. 构建项目

确保仍位于仓库根目录，并且已执行 `conda activate` 与 `source`：

```bash
export MAX_JOBS=8
ADANGEL_BUILD_CUDA=1 python -m pip install -v -e . --no-build-isolation --no-deps
```

`MAX_JOBS=8` 是保守起点；如果服务器内存有限可降低到 4。构建命令显式生成
`compute_120a/sm_120a`，并拒绝错误的 torch CUDA build 或 CUTLASS commit。

构建后执行：

```bash
python -m adangel doctor
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

O0/O1 正式后端已经实现并启用各自的独立 capability；O2 capability 仍关闭。因此
`python -m adangel doctor --require-native` 和完整正式 benchmark 目前仍应拒绝运行；
这不是 Conda 配置失败，也不能用 reference 后端绕过。

先独立验收 O0/O1（不需要模型和 trace）：

```bash
python -c "import torch; import adangel._sm120 as m; print(dict(m.capabilities()))"
python scripts/validate_o0.py
python -m pytest tests/integration/test_sm120_o0.py -q --run-sm120
python scripts/validate_o1.py
python -m pytest tests/integration/test_sm120_o1.py -q --run-sm120
```

预期 `o0_fp16_tc=true`、`o1_int8_tc=true`，两个验证脚本都输出 `passed=true`。
O0 会核对反量化、FP32 输出和 cuBLASLt HMMA；O1 会核对精确 E2M1→INT8 映射、
fused tiled signed-INT8 WMMA、逐 K32 INT32 partial/FP32 寄存器累加、无全局 partial
buffer 以及单次最终输出写回。两者都会验证四种计时模式。若源码是在 editable install
之后更新的，必须先重新执行本节构建命令；只重启 Python 不会重新编译 `.so`。

专项小矩阵通过后，使用 `4096^3` 对 O1 再做一次正式形状验收：

```bash
python scripts/validate_o1.py --m 4096 --n 4096 --k 4096 --warmup 5 --repeats 10 \
  | tee reports/o1_4096_fused_validation.json
```

O2 adapter 完成后再执行完整验收：

```bash
python -m adangel doctor --require-native
python -m pytest tests/integration -q --run-sm120
python -m adangel verify-layout --require-native
bash scripts/audit_instructions.sh build reports/audit
```

## 8. 外部 trace 与正式实验

本流程不下载 Llama-2-7B。把外部准备好的 24 个 FP16 trace 文件复制到
`data/raw/llama2_7b_prefill/`，然后按 README 的“生成公共输入”“正式运行”和“四表四图”
步骤执行。不要安装 `requirements/server-trace-optional.txt`，除非以后明确需要服务器自行
采集模型 trace。

## 9. 保存可复现环境快照

所有依赖安装完成后，在每次正式实验前保存一次环境：

```bash
mkdir -p runs/environment_snapshot
conda env export --name adangel-sm120 > runs/environment_snapshot/conda-environment.yml
python -m pip freeze > runs/environment_snapshot/pip-freeze.txt
python scripts/check_server_prereqs.py > runs/environment_snapshot/prerequisites.json
nvidia-smi -q > runs/environment_snapshot/nvidia-smi.txt
nvcc --version > runs/environment_snapshot/nvcc.txt
```

正式 runner 还会在每个 run 内生成 `environment.json`。这样既保留仓库中预期版本，也保留
服务器当次运行的真实驱动、频率状态和 Python 包解析结果。

## 10. 常见问题

- `nvidia-smi` 显示 CUDA 13.2、`nvcc` 显示 12.8：正常，前者是驱动能力上限，后者才是
  当前编译 Toolkit。
- `which nvcc` 不是 `/usr/local/cuda-12.8/bin/nvcc`：重新 source
  `scripts/activate_server_env.sh`，再检查 `PATH`。
- `torch.version.cuda` 不是 12.8：卸载错误的 torch wheel，并严格使用 cu128 专用索引重新
  安装。
- `torch.cuda.is_available()` 为 false：先确认 `nvidia-smi` 正常，再检查当前 Conda 环境和
  torch wheel；不要通过安装另一个系统驱动来掩盖问题。
- `check_server_prereqs.py` 的 Ubuntu 项失败：以 `cat /etc/os-release` 为准；如果服务器并非
  24.04，应先更新机器配置与实验环境记录，而不是跳过检查。
- 编译进程被杀：通常是主机内存不足，降低 `MAX_JOBS`，与 RTX 5090 显存大小无关。
- `Persistence-M: Off`：不影响环境构建。正式计时前可在管理员许可下启用 persistence
  mode，并确保 GPU 没有其他计算任务；无权限时如实记录即可。

# 双服务器 Trace 采集指南

本指南用于在存有本地 Llama-2-7B 的模型服务器采集真实 FP16 prefill trace，再把 trace
传到 RTX 5090 服务器完成公共量化准备和 O0/O1/O2 实验。模型文件不需要传到 5090。

这条数据链路只新增 Python 采集、校验和准备步骤，不修改 `csrc/sm120/`、CUTLASS、
后端能力位或指令审计逻辑，也不要求重新编译已经通过验收的 O0/O1/O2 扩展。

## 1. 两台服务器的职责

| 服务器 | 工作 | 不执行 |
|---|---|---|
| 模型服务器（H100 80GB） | 加载本地 Llama-2-7B、下载或读取 WikiText、采集 24 个 FP16 样本、生成 manifest | SM120 构建、CUTLASS 获取、O0/O1/O2 benchmark |
| RTX 5090 服务器 | 校验传输完整性、生成公共 INT8/MXFP4 输入、运行 O0/O1/O2 | 加载或下载 Llama-2-7B |

数据流为：

```text
tensor_hook_cu128
  -> 克隆为 adangel-trace
  -> Llama-2-7B FP16 + WikiText-2
  -> data/raw/llama2_7b_prefill
  -> rsync
  -> RTX 5090 二次校验
  -> data/prepared/llama2_7b_prefill
```

## 2. 模型服务器环境

克隆已经可工作的环境，避免修改原 `tensor_hook` 项目：

```bash
conda create --name adangel-trace --clone tensor_hook_cu128
conda activate adangel-trace

cd /path/to/ADAngel_oyzl
python -m pip install -e . --no-deps
```

这里不要设置 `ADANGEL_BUILD_CUDA=1`，不要运行 `fetch_cutlass.sh`，也不要运行
`audit_instructions.sh`。采集器是纯 Python 路径，不导入 SM120 扩展。

先保存环境信息：

```bash
mkdir -p runs/trace_environment
python - <<'PY' | tee runs/trace_environment/model_server_packages.json
import json
import platform
import torch
import transformers
import datasets
import accelerate
import yaml
import safetensors
import sentencepiece
import google.protobuf

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_total_memory_bytes": (
        torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available() else None
    ),
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "pyyaml": yaml.__version__,
    "safetensors": safetensors.__version__,
    "sentencepiece_import": True,
    "protobuf": google.protobuf.__version__,
}, indent=2))
PY

nvidia-smi > runs/trace_environment/nvidia-smi.txt
conda env export --name adangel-trace > runs/trace_environment/conda-environment.yml
python -m pip freeze > runs/trace_environment/pip-freeze.txt
```

要求 Python 3.10、CUDA 12.8 可用、GPU 为 H100 80GB，Transformers 主版本为 4，并且上述包均可导入。
复用环境已经满足时不要主动升级或降级。只有出现缺包时，先安装对应缺失项；也可以使用：

```bash
python -m pip install -r requirements/server-trace-optional.txt
```

Llama 模型目录没有预生成 fast-tokenizer 文件时，Transformers 会从 SentencePiece
转换 tokenizer；该路径同时依赖 `protobuf`。若出现
`LlamaConverter requires the protobuf library`，执行：

```bash
python -m pip install protobuf==5.29.3
```

该 requirements 使用兼容范围而非覆盖已有环境的全部精确版本。WikiText 第一次读取需要
Hugging Face 网络访问或已存在的本地 datasets cache；模型则始终通过
`local_files_only=True` 从 `--model` 指定的绝对路径加载。

## 3. 固定采集配置

`configs/trace/llama2_7b_prefill.yaml` 固定：

- WikiText 数据集 `Salesforce/wikitext`、子集 `wikitext-2-raw-v1`、train split；
- revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
- seed `20250805`；
- 4095 个连续语料 token，开头添加一个 Llama BOS，共 4096 个有效 token；
- batch 1、FP16、SDPA、`use_cache=False`；
- 层 `0,6,12,18,24,31`；
- 每层 `q_proj/k_proj/v_proj/o_proj`，共 24 个样本。

采集器会先验证 Llama 配置：hidden size 4096、32 层、32 个 attention head、32 个 KV head，
最大位置长度至少 4096，且所有被采集 projection 的权重均为 `[4096,4096]`。

## 4. 正式采集

模型路径必须是模型服务器上的绝对本地目录：

```bash
python scripts/collect_trace.py \
  --model /absolute/path/to/Llama-2-7b-hf \
  --output data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml \
  --device cuda:0
```

采集器只调用内部 `model.model`，不会计算 LM head logits。pre-hook 直接检查 Linear 输入和
权重原本就是 FP16；如果实际推理产生 BF16 或 FP32，会失败而不是事后转换。每个 hook 将
activation 和 weight 复制到 CPU contiguous tensor，原子写入独立样本文件。

为防止覆盖已有数据，目标目录已存在时命令会拒绝运行。采集中断时可能留下名称形如
`.llama2_7b_prefill.collecting-<uuid>` 的临时目录；确认没有有效任务仍在使用后，可手工检查
并处理，不会自动覆盖正式目录。

## 5. 模型服务器验收

```bash
python scripts/validate_raw_trace.py \
  --input data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml
```

完整校验包括：

- manifest 格式、固定数据集 revision、输入 token SHA-256 和环境 provenance；
- 恰好 24 个样本，无缺失或额外 `.pt`；
- 每个文件的 SHA-256、身份、shape 和 dtype；
- A/W 都是 contiguous FP16 `[4096,4096]` 且无 NaN/Inf；
- 同一层 q/k/v 的 Linear 输入逐元素一致。

成功时输出 JSON，并且 `"passed": true`。若只需要快速检查 manifest 与文件哈希，可额外用
`--metadata-only`；正式传输前后必须运行不带该选项的完整校验。

## 6. 传输到 RTX 5090

在模型服务器执行：

```bash
rsync -avP --partial \
  data/raw/llama2_7b_prefill/ \
  zlouyang@<5090服务器地址>:/home/zlouyang/oyzl/ADAngel_oyzl/data/raw/llama2_7b_prefill/
```

末尾的两个斜杠表示复制目录内容。原始 trace 约 1.5 GiB；`--partial` 允许断线后续传。
不要只复制 `.pt` 而漏掉 `trace_manifest.json`。

## 7. RTX 5090 二次校验与准备

```bash
cd /home/zlouyang/oyzl/ADAngel_oyzl
conda activate adangel-sm120
source scripts/activate_server_env.sh

python scripts/validate_raw_trace.py \
  --input data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml

python scripts/prepare_trace.py \
  --input data/raw/llama2_7b_prefill \
  --output data/prepared/llama2_7b_prefill \
  --config configs/experiment/o0_o1_o2_4096.yaml \
  --trace-config configs/trace/llama2_7b_prefill.yaml
```

准备步骤对每个样本生成：

```text
A_int8   [4096,4096] int8
A_scale  [4096]      fp32
W_mxfp4  [4096,2048] packed uint8
W_scale  [4096,128]  UE8M0 uint8
```

公共准备耗时不进入 O0/O1/O2 结果。prepared `manifest.json` 会嵌入原始 manifest 的
SHA-256、数据集 revision、token 起点和 input ID hash、模型文件 hash、采集环境及推理设置。

如果 `data/prepared/llama2_7b_prefill` 已非空，准备命令会拒绝覆盖。需要重做时请先人工
确认旧目录是否应保留并使用新的输出目录，避免误删历史实验数据。

## 8. 后端与重编译边界

仅同步本次 trace 采集相关的 Python、YAML 和文档改动时：

- 不需要重新执行 `ADANGEL_BUILD_CUDA=1 python -m pip install ...`；
- 不需要重新获取 CUTLASS；
- 不需要重新运行 PTX/SASS 指令审计；
- 已构建的 `python/adangel/_sm120*.so` 可以继续使用。

若仓库以 editable 模式安装，Python 脚本改动会直接生效。只有之后实际修改
`csrc/sm120/`、CUDA 构建配置或原生扩展接口时，才需要重新编译并重新审计。

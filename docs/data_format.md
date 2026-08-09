# 数据格式

项目有两级数据：模型服务器产生的原始 FP16 trace，以及 RTX 5090 上一次性生成的公共
INT8/MXFP4 prepared 输入。正式 runner 只读取 prepared 数据，但 prepared manifest 必须保留
原始 trace 的 provenance。

## 1. 原始 FP16 trace

目录固定为：

```text
data/raw/llama2_7b_prefill/
  trace_manifest.json
  layer_00_q_proj.pt
  layer_00_k_proj.pt
  layer_00_v_proj.pt
  layer_00_o_proj.pt
  ...
  layer_31_o_proj.pt
```

层为 `0,6,12,18,24,31`，projection 顺序为
`q_proj,k_proj,v_proj,o_proj`，因此恰好有 24 个 `.pt` 文件。目录中出现缺失或额外
`.pt` 文件都会使正式校验失败。

### 1.1 样本文件

每个 `layer_XX_<projection>.pt` 是 `torch.save` 写入的字典：

```python
{
    "sample_id": "layer_00_q_proj",
    "layer": 0,
    "projection": "q_proj",
    "activation_fp16": Tensor[4096, 4096],  # contiguous torch.float16
    "weight_fp16": Tensor[4096, 4096],      # contiguous torch.float16
}
```

`activation_fp16` 是该 Linear 的输入去掉 batch 维后的矩阵 A；
`weight_fp16` 是 PyTorch Linear 的权重 W。实验计算统一解释为
`Y = A @ W.T`，所以文件中不预先转置权重。

所有 tensor 必须原生为 FP16、CPU contiguous、无 NaN/Inf。同一层的 q/k/v projection 接收
相同 hidden state，因此三份 activation 必须逐元素完全一致；o_proj 输入不同，不做该约束。

### 1.2 trace_manifest.json

manifest 顶层格式为：

```text
version = 1
format  = adangel-raw-fp16-prefill-trace
```

关键字段：

| 字段 | 内容 |
|---|---|
| `trace` | batch、4096 token、FP16、层/projection 集合和期望 shape |
| `dataset` | WikiText ID、subset、split、锁定 revision、fingerprint、语料 hash |
| `tokenization` | seed、随机起点、BOS、完整 4096 input IDs 及 SHA-256 |
| `model` | Llama 配置和本地 config/tokenizer/权重分片 SHA-256 |
| `runtime` | Python、PyTorch、CUDA、GPU、driver 和采集依赖版本 |
| `inference` | FP16、SDPA、无 KV cache、只运行 base model、本地模型加载 |
| `environment_source` | 克隆环境来源项目和 commit |
| `samples` | 24 个文件的 identity、shape、dtype 和 SHA-256 |

input ID hash 对无空格的 JSON 整数数组（例如 `[1,42,...]`）计算 SHA-256。manifest
同时保存完整 input IDs，校验器会按同一 canonical JSON 规则重新计算，而不是只相信记录值。

### 1.3 原始数据校验

模型服务器和 RTX 5090 都运行：

```bash
python scripts/validate_raw_trace.py \
  --input data/raw/llama2_7b_prefill \
  --config configs/trace/llama2_7b_prefill.yaml
```

正式校验会加载 tensor，约需要读取整套约 1.5 GiB 数据。`--metadata-only` 只适合快速诊断；
它仍校验 manifest、文件集合和 SHA-256，但不检查 tensor 内容与 q/k/v 一致性。

## 2. 公共 prepared 输入

在 RTX 5090 执行 `scripts/prepare_trace.py` 后得到：

```text
data/prepared/llama2_7b_prefill/
  manifest.json
  layer_00_q_proj.pt
  ...
  layer_31_o_proj.pt
```

每个文件内容为：

```python
{
    "sample_id": "layer_00_q_proj",
    "A_int8": Tensor[4096, 4096],    # torch.int8
    "A_scale": Tensor[4096],         # torch.float32
    "W_mxfp4": Tensor[4096, 2048],   # packed E2M1, torch.uint8
    "W_scale": Tensor[4096, 128],    # UE8M0, torch.uint8
    "shape": [4096, 4096, 4096],
}
```

公共量化规则：

- 激活：逐行对称 INT8，范围 `[-127,127]`，零行 scale 为 1；
- 权重：E2M1 + UE8M0，K32 分组；
- rounding：round-to-nearest, ties-to-even；
- packed MXFP4 的偶数 K 在低 nibble，奇数 K 在高 nibble；
- UE8M0 code 255 禁止出现。

prepared `manifest.json` 顶层格式为：

```text
version = 2
format  = adangel-prepared-mxfp4-k32
```

除 24 个样本的 SHA-256 外，还包含矩阵 shape、dtype、量化规则和 `source_trace`。
`source_trace.manifest_sha256` 指向原始 `trace_manifest.json`，并复制数据集、tokenization、
模型、runtime、inference 和环境来源信息，从而能从任意实验 run 追溯到采集输入。

公共 FP16 -> INT8/MXFP4 准备只执行一次，其耗时不属于 O0/O1/O2 的量化或反量化开销。

## 3. 正式 run 结果

每次 run 目录至少包含：

```text
runs/<run_id>/
  config.yaml
  environment.json
  results.jsonl
  summary.json
```

每条 `results.jsonl` 记录包括样本 ID、variant、计时模式、阶段延迟、吞吐、MSE 和运行环境。
O0 FP32 输出作为唯一 MSE 主参考；O1/O2 的 reduction 转为 FP64 计算。结果与报告格式详见
`docs/experiment_protocol.md`。

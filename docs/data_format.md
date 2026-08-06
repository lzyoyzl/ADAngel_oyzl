# 数据格式

## 原始外部 trace

本轮不下载模型。每个样本由外部产生并保存为一个 `.pt` 文件，文件名从
`layer_00_q_proj.pt` 到 `layer_31_o_proj.pt`，内容为：

```python
{
  "sample_id": "layer_00_q_proj",
  "layer": 0,
  "projection": "q_proj",
  "activation_fp16": Tensor[4096,4096],  # torch.float16, contiguous
  "weight_fp16": Tensor[4096,4096],      # torch.float16, contiguous
}
```

准备程序严格要求层 `0,6,12,18,24,31` 与 projection
`q_proj,k_proj,v_proj,o_proj` 的笛卡尔积，共 24 个文件；缺失、额外、命名/元数据不一致、
非 FP16、非 contiguous、错误 shape 或 NaN/Inf 均直接失败。

## 公共 prepared 样本

```python
{
  "sample_id": str,
  "A_int8": Tensor[4096,4096],
  "A_scale": Tensor[4096],
  "W_mxfp4": Tensor[4096,2048],
  "W_scale": Tensor[4096,128],
  "shape": [4096,4096,4096],
}
```

UE8M0 `0..254` 解码为 `2**(code-127)`，`255` 是 NaN，不允许出现在正式样本。全零
block 因 E8M0 没有零编码，固定使用 code 127（scale=1）且所有 E2M1 为零。scale 指数低于/
高于范围时分别饱和到 code 0/254，并由单元测试覆盖。

## manifest v2

每个 prepared 目录包含 `manifest.json`：

```json
{
  "version": 2,
  "format": "adangel-prepared-mxfp4-k32",
  "matrix_shape": [4096, 4096, 4096],
  "dtypes": {
    "A_int8": "torch.int8",
    "A_scale": "torch.float32",
    "W_mxfp4": "torch.uint8",
    "W_scale": "torch.uint8"
  },
  "quantization": {
    "activation": "int8_symmetric_per_row",
    "weight": "mxfp4_e2m1_ue8m0_k32",
    "rounding": "round_ties_to_even"
  },
  "trace": {
    "source": "external_fp16_prefill_trace",
    "batch_size": 1,
    "valid_tokens": 4096,
    "layers": [0, 6, 12, 18, 24, 31],
    "projections": ["q_proj", "k_proj", "v_proj", "o_proj"]
  },
  "samples": [
    {
      "sample_id": "layer_00_q_proj",
      "file": "layer_00_q_proj.pt",
      "sha256": "<64 lowercase hex chars>",
      "shape": [4096, 4096, 4096],
      "layer": 0,
      "projection": "q_proj"
    }
  ]
}
```

正式 runner 在创建 run 目录前验证 manifest 契约，并重新计算所有 24 个 `.pt` 文件的
SHA-256；内容被修改或文件集合不完整时不会开始实验。

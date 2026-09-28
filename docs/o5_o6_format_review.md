# O5/O6 格式与数据核对记录

核对日期：2026-09-28。状态：**下述格式与 F 选择已获用户确认；正式数据处理口径待澄清**。
用户随后指定复用 A100 已有 prepared 数据，不再要求补充原始 FP16 trace。
“不需要重新生成 NVFP4/MXFP8/HiF4/FP6”是指不另存中间文件，还是跳过这些
格式的数值量化，尚待确认。两者实验含义不同，不能默认等价。
本文不是 O5/O6 完成报告；不更改 O0–O4 默认后端，不复制原始 trace。

## 1. 已确认：精度项目与 O3 的 E2M1→Q4 数值语义一致

核对的版本及入口：

- 精度项目 `/root/llama2-7-Acc`：`84296780456ce9ffb749487a2493f684cf71906b`，
  `src/quantization/fake_quant.py` 的 `_cast_e2m1_to_fixed()` 和
  `fake_quant_mxfp4_to_fixed()`；该文件核对时无未提交修改。
- 本项目：`6e03a23cf8f9924581f1e296af8d8d738bd49aec`，
  `include/adangel/data_types.cuh` 的 `e2m1_to_q4()`，以及
  `python/adangel/reference/o3.py` 的 G128 scale 应用。

共同语义：先取得已经量化的局部 E2M1 数值 v，保留原 group scale s，再进行：

```text
L = 2^(Q-1) - 1
q = clamp(RNE(v * 2^F), -L, L)
reconstructed = q * 2^(-F) * s
```

RNE 指最近舍入、正中间取偶数。现有 E2M1 转换规定 F=Q-4。
Q4/Q5/Q6 分别使用 F=0/1/2。Q5/Q6 不增加对已量化 E2M1 的数值误差，
不意味着 FP16→E2M1 无误差。

以纯标量检查穷举了 16 种 E2M1 编码；RNE 公式与 CUDA Q4 查表所得整数
16/16 一致。正幅值映射为 `{0,0.5,1,1.5,2,3,4,6} → {0,0,1,2,2,3,4,6}`。
这是转换语义检查，不是新 kernel 的 GPU 验收。

精度项目的 code 是符号—幅值编码；O3 使用补码 S4。二者数值相同，但负数
编码不相同，负零在整数后端归零。不能将精度参考的 code 原样输入 S4 MMA。

## 2. 已核实的格式资料

| 名称 | 元素 | 原始 scale / 分组 | G128 实验需要明确的变化 |
|---|---|---|---|
| NVFP4 | E2M1 | 每 16 元素 E4M3 scale，另有 tensor 级 FP32 scale | 保留两级 scale，将一维组长改为 128 |
| MXFP8 | E4M3 或 E5M2 | 每 32 元素 E8M0 / UE8M0 scale | 选择元素编码，将组长改为 128 |
| HiF4 | 符号—幅值 S1P2 | G64 的 E6M2 基础 scale + 每 8、每 4 元素的两级 1-bit 微指数 | 需要显式扩展层级，不能忽略微指数 |
| NVIDIA FP6 | E2M3 或 E3M2 | 元素编码本身不唯一规定 block scale | “NVFP6”不是足以确定编码与 scale 的完整契约 |

来源：[NVIDIA NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)、
[NVIDIA MXFP8](https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html)、
[CUDA 12.8 FP6](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html)、
[HiF4 格式论文](https://arxiv.org/html/2602.11287v1#S2)。
这里的标准格式与实验扩展必须分开命名；G128 不等于上述原版分组。

### HiF4 参考实现与转换边界

作者的 [HiFloat4 仓库](https://github.com/global-computing-consortium/HiFloat4)
包含 Python/CUDA 伪量化参考；核对固定版本为
`6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4`。这不是可直接调用的 A100 INT4 GEMM。

其数值结构为 `x = s_E6M2 * 2^(e8+e4) * v_S1P2`。
若后端每 G128 只保留一个权重 scale，转定点前必须先把
`2^(e8+e4)` 合入局部值。只转换 S1P2 而丢弃微指数会改变数值。
合入后局部最大幅值为 7；以 F=0 转 Q4 会对其中的分数产生额外舍入。

[参考 Python 源码](https://github.com/global-computing-consortium/HiFloat4/blob/6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4/hif4_gpu/HiF4_NVFP4_v14f16.py)
虽然暴露 G 参数，内部子组数量仍按 64 编写；不能仅传 G=128。
已确认的扩展是保留 8/4 元素的微指数粒度，改为每 128 元素一个 E6M2，
对应 16 个二级微指数、32 个三级微指数，标记为 HiF4-like-G128。
该扩展已有独立标量及 Torch 参考，仍需原生转换、GEMM 和正式数据验收。

参考初始 HiF4 量化含 BF16 中间舍入，S1P2 部分使用幅值 half-up。
这与随后独立的“源格式→定点 RNE”是不同步骤；不能将整个参考转换
笼统描述为全部 RNE，或在移植时悄悄改变中间舍入。

## 3. 已确认：源格式到统一整数接口

保留外部 scale、对局部数值 RNE 的原则不变，但 F=Q-4 是 E2M1 特定选择，
不能无条件套用 FP8/FP6。用户已确认以下选择：

| 路径 | 局部源值 | Q | F | 后端有效 scale |
|---|---|---:|---:|---|
| O5 W | E2M1 | 4 | 0 | E4M3 block scale × FP32 tensor scale |
| O5 A | E4M3 | 8 | -2 | UE8M0 block scale × 4 |
| O6 W | 合并微指数后的 HiF4 局部值 | 4 | 0 | E6M2 基础 scale |
| O6 A | E2M3 | 6 | 2 | 源有效 scale ÷ 4 |

F 的建议按完整有限数值范围适配对称整数区间，不按每组原始 FP16 重新拟合：
`F = floor(log2((2^(Q-1)-1) / max_abs_local))`。
例如 E4M3 最大幅值 448，使用 F=-2 得到整数 112；若机械使用 F=8-4=4，
则局部值大于 127/16 就会饱和。Q6 的 E2M3 转换也不能假称无误差。

对于 O6 中暂称的 NVFP6，已确认采用 **NV-style FP6-G128**：
E2M3 payload + G128 E4M3 scale + tensor FP32 scale。
这是本实验的变体定义，不能称为已经核实的 NVIDIA 标准格式。
若选 E2M3 + UE8M0，则属于 MXFP6 路线，须先取得用户同意再改变实验命名。
[NVIDIA 的 MX 格式表](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)
提供了 MXFP6 的明确元素与 scale 定义。

Q6 转 INT8 时必须符号扩展，再按 low U4 + 16×high S4 分解。
这与 CUDA FP6 原始存储中的零填充不是同一操作。
参考实现位于 `python/adangel/quantization/mixed_formats.py`。
源格式到整数的 scale 补偿为 FP32，不将参考 Torch 转换耗时当成原生性能。

## 4. 数据来源与当前待确认边界

2026-09-28 通过 A100 SSH 只读检查 `/home/zlouyang`：按原始 manifest 名称
及数据目录查找，未找到标准原始 FP16 trace；项目 `data/raw/` 为空。
已有 `data/prepared/llama2_7b_prefill_o0_o4/manifest.json`，格式为
`adangel-prepared-mxfp4-k32-g128-q4`，记录的是 INT8/MXFP4/Q4 及 scale。
其中 source_trace provenance 不表示原始 FP16 tensor 仍在本机。
结论仅限上述检查范围，不声称检查了服务器所有挂载点。

历史 A100 O0/O1/O3 运行直接读取上述 prepared 数据；其 provenance 源于
H100 上采集的 24 个 Llama-2-7B FP16 prefill 样本。现存 prepared 本身已量化，
不是原始 FP16。24 个文件均未包含 activation_fp16 / weight_fp16 张量。
当前 prepared manifest SHA-256：
`05849422f6d8ad18e7c3468ff6af549d33ce86eeb9590af7388ab3452d26881c`；
所记原始 trace manifest SHA-256：
`4ff05585d91f8940f20328b140c637ac948d0dbe47f1506db0cb5d839c9c3db0`。

用户已授权改用 A100 现有数据，不再以传输原始 trace 作为前提。
若从 A_int8/W_mxfp4 反量化后再进行源格式量化，须标记为二次量化实验；
不能冒充 FP16 直接量化实验。若完全跳过源格式量化，只能称为整数后端
测试，不能据此宣称验证了 NVFP4/MXFP8/HiF4/FP6 精度。
该区别已向用户提问，当前只推进与两种数据来源独立的编码及整数接口验证。

正式验收仍需确定后的 24 样本、MSE vs 同源 O0、四种计时、配对 O0 性能
以及指令/内存安全验证。不用合成数据冒充真实数据结果。
现有 [共用整数 core 证据](evidence/a100_split_grouped_v1/README.md)
不替代这些验收；目标仍是完整 O5/O6，而不是只交付整数 core。

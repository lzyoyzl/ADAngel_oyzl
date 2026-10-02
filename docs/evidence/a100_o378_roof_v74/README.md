# v74：全 K scale 提前乘入输入——精确范围检查不通过

本轮是**可行性检查，不是新的 kernel 或性能版本**。目的是判断能否在准备阶段吸收
全部 G128 scale，让 GEMM 不再逐输出乘 A/W 整数 factor，同时保留 INT8 激活、INT4 权重、
两路原生 INT4 MMA、现有量化和输出语义。没有修改正式默认、O3或5090。

## 方法

对每一行激活、每一列权重独立检查。将有效组 scale 精确写成 `m[g] × 2^e[g]`，
选择非零 payload/scale 组的最小指数 h，形成任意精度整数：

```text
v[g,k] = payload[g,k] × m[g] × 2^(e[g]−h)
d = gcd(这一行所有 v)
new_payload = v / d
new_base = 原公共因子 × 2^h × d
```

最大公因数全部移入公共 base，使整数范围尽可能小。全零组不强迫使用更小锚点。
在“一行一个正公共 base、无 zero point”的精确表示约束下，这是最小整数表示；
检查只判断范围，**尚未要求 new_base 能无误差表示为 FP32，因此是乐观筛选**。
O8 原 Q6 激活甚至允许扩到现有计算路径可接受的 INT8，仍未通过。

A100 使用原始 FP16 trace，按现有 `quantize_source`、`to_fixed_reference` 生成源格式和 payload；
不从已经舍入的输出反推，也不改量化参数。检查首层 q/k/v/o 四个真实样本，4096³、32个G128。
每组保留 scale code、payload 最小/最大值及精确 GCD；Python 任意精度计算避免范围检查自身溢出。

## 结果

|后端/操作数|需要保留的整数位宽|吸收全部 scale 后最小有符号位宽范围|符合原位宽的行数（四样本合计）|
|---|---:|---:|---:|
|O7 激活|8|11–17|0 / 16,384|
|O7 权重|4|8–13|0 / 16,384|
|O8 激活|8（放宽自Q6）|13–18|0 / 16,384|
|O8 权重|4|7–12|0 / 16,384|

各样本 A 的64个行tile、W的32个列tile也全部不符合原位宽。
表中位宽是**吸收 scale 后的内部整数范围**，不是改变了源格式的量化位数。
同层 q/k/v 的激活相同，这四个投影不是四个独立输入分布；不能外推为24样本全部检查。

**停止这一直接全量吸收方案，不进入CUDA实现、不扩展24样本计时。** 强行压回原位宽会增加量化误差；
保留精确值则需要更宽整数或额外计算，不能当成免费的 scale 消除。
这不否定部分吸收、A/W联合重新分配K相关因子等其他算法，但它们没有在本轮获得性能证据。

本轮没有新 GEMM/Conversion/Cold/Steady 加速比，也没有新 MSE。当前最佳仍是既有全K GEMM候选
与v73行级融合准备；不能把“提前否定一条路线”写成性能提升。

## 证据与复现

源码本地提交并push后，A100 fetch/ff-only到 `12167353b765b5b030df1773a12f5e7346d9ca48`。
环境为A100-PCIE-40GB、PyTorch2.7.1+cu128。源模型trace/准备manifest与源格式、payload身份记录在
[summary.json](reports/o378_roof_v74_input_fold_screen/summary.json)及16份详细JSON中。
传输归档SHA256：`6a541dfc4aa13a0a9163996661989266377e6c942b2bab1b8af84dc49fc66328`。
离线证据测试重算全部65,536行，并校验全部详细文件SHA与直方图；
算法测试包含300组随机逐元素对照、符号边界、零组、非法scale和tile门槛。

```bash
python scripts/inspect_o78_input_factor_folding.py \
  --output reports/v74_recheck --samples 4
python -m pytest tests/unit/test_o78_input_factor_folding.py \
  tests/unit/test_roof_v74_evidence.py -q
```

运行目录必须为尚不存在的项目内路径。没有执行新的GEMM或sanitizer，亦不据此宣称它们通过。

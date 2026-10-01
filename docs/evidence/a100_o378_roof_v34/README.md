# v34：整数源编码转换与目标布局遍历（隔离转换验收完成）

本轮是独立conversion-only探针，不改变正式默认、现有GEMM或5090后端。
它服务于降低O7/O8端到端开销；不会把转换收益宣称为接近纯GEMM容量上界的收益。

四个对照：

|ID|实现|目的|
|---|---|---|
|0|现有自然布局转换 + G128-major重排|当前最佳GEMM使用的转换基线|
|1|v25原解码/查表 + 融合输出重排|既有融合候选，不能忽略其负结果|
|2|整数RNE解码 + 既有flat遍历|隔离payload解码改动|
|3|整数RNE解码 + 每CTA处理一个G128的四行|减少重排地址计算，目标数据连续写入|

E4M3→Q8仍F=-2，E2M3→Q6仍F=2，HiF4→Q4仍F=0并先合并微指数。
E2M1→Q4的原16编码映射用一个32bit寄存器LUT表示。
有限源编码的局部值均为二进制有理数，以整数商/余数实现相同RNE；
所有有效scale的FP32解码及乘法顺序完全保留。没有跨G128合并，没有magic-bias。

host复用原始源格式检查，限制索引乘积和二维grid范围。
所有显存与CUDA Event在预热前分配，内层100次摊销；不计入源格式公共量化。
独立入口`_benchmark_mixed_conversion_probe`不接入正式实验调度。

## 24个真实样本结果

原始FP16 trace重新产生相同源格式；同binary、单轮循环换序，warmup50、repeats200、inner100。
总计384条转换记录，所有计时CV<3%，不筛离群。变化为逐样本配对吞吐变化。

|转换阶段|原路径0 median ms|选中路径 / median ms|配对吞吐提升|描述性speedup 95%区间|
|---|---:|---:|---:|---|
|O7-W，NVFP4|0.085617|3 / **0.037033**|**+132.30%**|[2.30489,2.33237]|
|O7-A，MXFP8|0.088745|1 / **0.081733**|**+8.67%**|[1.08467,1.09327]|
|O8-W，HiF4|0.062700|3 / **0.054892**|**+13.95%**|[1.13539,1.14792]|
|O8-A，FP6实验变体|0.088699|3 / **0.074045**|**+19.80%**|[1.19475,1.20221]|

完整消融的配对吞吐变化（对照均为0；NVFP4原本就是整数查表，并非所有原转换都做浮点解码）：

|格式|1：旧融合|2：整数flat|3：整数目标布局|
|---|---:|---:|---:|
|NVFP4|+7.40%|+49.91%|+132.30%|
|MXFP8|+8.67%|−9.71%|+0.41%|
|HiF4|−13.49%|−18.17%|+13.95%|
|FP6实验变体|+3.24%|+5.96%|+19.80%|

**整数化不是统一赢家；消除重排和地址计算要与格式解码共同考虑。**
NVFP4利用寄存器LUT收益明显；HiF4单改整数解码更慢，改目标布局后才获益；
MXFP8仍选择旧浮点融合。这里没有测端到端，不把这些比例当作GEMM/cold/steady收益。
资源文件可见旧NVFP4融合转换有8B stack，新寄存器LUT为0；这是移除动态索引线程局部查表，
不能笼统归因于“消除浮点转换”，也不能把这种局部表直接等同于寄存器压力造成的spill。
三个合成初筛round和所有负结果都保留。24个样本来自同一trace，区间仅作描述性统计。

## 正确性、审计与限制

- 524项GPU编码/scale/布局检查、24项非法输入检查通过，包含非默认stream、尾行和非偶数group数。
- 有限memcheck/synccheck各108项、24项拒绝检查通过，均0 errors；不是全输入空间安全证明。
- 24样本×2后端×4转换方案，共192项最终输出检查与当前GEMM59逐位一致。
  相对旧结果MSE=0；O7/O5 median MSE仍0.005536172426666，O8/O6仍0.004411084948645。
- CUDA12.8的8个转换kernel为14～22寄存器、零stack/local。4个目标布局实例无F2I；
  flat控制的通用整数地址除法被编译为I2F/MUFU.RCP/F2I.U32.TRUNC，不能说整个flat kernel完全无浮点指令。
  原审计对所有实例一律禁止F2I过宽，初次失败结果保留；修正后按明确遍历类别检查。
- 12个旧正式GEMM和148个旧候选的编码SASS不变，原生双INT4审计通过（既有GEMM spill如实保留）。
  另4个非目标mixed-binary函数codegen变化，完整mixed格式回归通过；全范围codegen报告保留passed=false。
- 只验证转换及通过现有prepared-core入口重跑输出；没有声称已测新路径的四模式。
  后续v35将上述固定格式选择接入同一个59，重新测直接端到端。

## 证据与复现

源码d94bbe5；runner0a22fc1；审计范围修正8c87765。
binary SHA-256：`605fd27299aa203f7fbcaaa309d509ceba0dcd01b2d4942c447cd7e9c46ffd36`。
40个文本证据经SHA和成员路径验证提取；归档SHA-256：
`2528f00fc1c4709c664cd442c1a6afbe7eb42c626a8fae7efd4bda5087cff291`。

主结果：`runs/o378_roof_v34_trace24/{results.jsonl,summary.json,output_mse.jsonl,environment.json}`。
8个新转换kernel的完整SASS在`reports/o378_roof_v34/conversion_kernels.sass`。
审计、codegen、预检、回归、sanitizer和合成初筛均在同目录保留。

```bash
python scripts/validate_integer_conversion_probe.py --output runs/conversion_preflight
python scripts/benchmark_integer_conversion_probe.py --output runs/conversion_trace24
python scripts/audit_integer_conversion_probe.py --output reports/conversion_audit
```

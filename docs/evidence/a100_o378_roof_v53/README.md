# v53：O7/O8向量转换，24样本独立筛选

**16元素/线程的候选胜过四种格式的既有标量最佳；尚未接入Cold/steady-state。**
本轮不修改GEMM59、不重编译正式扩展、不切换默认，不把转换收益当作GEMM收益。

## 实现与口径

三个固定策略：0为现有最佳（MXFP8用浮点融合，其余三格式用整数目标布局转换）；
1为8元素/线程；2为16元素/线程。候选在寄存器中批量解码/packing，向量加载和写回，
直接产生G128-major payload和scale；HiF4每四元素复用一次micro-scale解码。
每CTA256线程，策略1/2分别覆盖16/32行的一组G128。

格式及RNE、有效scale乘法顺序均不变。MXFP8仍保留浮点RNE，不能称为统一整数解码。
这是向量宽度、线程分工与索引/元数据摊销的组合优化，不是仅替换一条load的单因素实验。
旧12个转换entry的编码SASS与v35逐条一致；新8个实例15–32寄存器、零stack/local/spill。

真实FP16 trace直接生成源格式；公共源格式量化在计时之外。
24样本×3轮，warmup50/repeats200/inner100，单stream原生CUDA Event；
按样本、格式、轮次循环换序，不锁频、不筛离群。预分配和验证在计时外。
此处每项仅包含一个operand转换，不是O7/O8 conversion-total，也不是端到端。

## 转换结果

|阶段|原最佳标量 ms|8元素 ms / 配对吞吐提升|16元素 ms / 配对吞吐提升|16元素speedup描述性95% CI|
|---|---:|---:|---:|---|
|O7-W，NVFP4|0.037069|0.018683 / +98.68%|**0.015585 / +137.94%**|[2.37705, 2.38214]|
|O7-A，MXFP8|0.081661|0.038231 / +113.09%|**0.032384 / +151.47%**|[2.50754, 2.51864]|
|O8-W，HiF4|0.054943|0.032581 / +68.47%|**0.030587 / +79.32%**|[1.79074, 1.79612]|
|O8-A，FP6实验变体|0.074081|0.048435 / +53.19%|**0.045732 / +62.19%**|[1.61879, 1.62708]|

每个样本先对三轮配对speedup取中位数，再跨24样本汇总，非两列median直接相除。
864条记录均保留：只有FP6/8元素的一条CV≥3%；策略2的288条记录均CV<3%。
同一trace的24样本相关，CI仅作描述性比较。不同计时模式不能相加median或跨run拼接。

## 输出与MSE

各格式每轮payload/scale与Python参考逐位一致。转换结果送入同一个GEMM59，
共144项输出检查（24样本×2后端×3策略）全部与当前最佳逐位一致。
诊断用逆布局在计时外，仅用于调用既有prepared-core接口；不隐藏真实转换成本。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|相对当前最佳|
|---|---:|---:|---|
|O7 / O5|0.005536172426666|0.005053635833762|逐位不变，输出差MSE=0|
|O8 / O6|0.004411084948645|0.004381379215302|逐位不变，输出差MSE=0|

preflight及memcheck/synccheck/racecheck各完成393项检查、18项非法输入拒绝。
覆盖有限payload编码、有效scale编码、HiF4微指数、尾行/奇数组、非默认stream；
三个sanitizer均0错误，racecheck也0警告。相关CPU回归394项通过；新增证据测试另核对原始数据。

## NCU：为什么更快

同一真实样本`layer_00_q_proj`，每格式策略0/2各捕获一次目标转换。
`--set full --cache-control all --clock-control none`，过滤正式转换symbol，跳过50次warmup。
下表是profiling诊断，缓存口径与批量Event不同，**不是正式性能验收时间**。

|格式|NCU ms：0→2|动态warp指令：0→2|指令减少|寄存器：0→2|
|---|---:|---:|---:|---:|
|NVFP4|0.048608→0.018528|14.418M→5.308M|63.18%|16→26|
|MXFP8|0.074336→0.028736|34.996M→12.091M|65.45%|20→32|
|HiF4|0.056384→0.031904|21.758M→10.879M|50.00%|16→30|
|FP6实验变体|0.068896→0.039232|23.251M→14.909M|35.88%|16→19|

每个SASS-PC的动态计数与raw总指令数相符，且核对了格式/模板参数，未匹配到GEMM/probe误入口。
向量版本的NVFP4/MXFP8/HiF4/FP6动态LDG数分别减少81.25%/90.00%/64.29%/87.50%。
这是load指令数，不是数据字节减少同样比例；宽load仍搬运所有必要数据。
scale写入每组只有一个线程负责，warp内有效lane比例变化也会改变warp指令计数。
因此不将每项减少都归因于单一微优化。

策略2的DRAM吞吐百分比依次30.63%/47.77%/19.68%/35.16%，不能说已达到HBM极限。
FP6仍保留数据相关分支，向量展开后的动态BRA约翻倍，抵消部分收益；
减少分支或使用小表是后续可评估方向，**本轮未测，不计为收益**。
主要工作仍是GEMM；这轮只为端到端回收转换成本，不改变约0.220347ms的GEMM容量下界。

## 身份、复现与下一步

- 构建/主测commit：`dcf0d3a01f1ea52f249088aaac8d58d468814c88`；profile脚本：`cbc0dc4`。
- 隔离库SHA256：`b7c47d2bc596cebf6e16bd75912f6ff142ddb6e4723bd066abf0e0e939cc563d`。
- 正式扩展前后未变：`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
- 下载归档SHA256：`4c032baea14ce0f6a1a82dacc6dafc9d0cbd9b3c75012712f6fb66ce25573ead`。
- 补充旧SASS相关12函数摘录SHA256：`17806897913e0b7c3f4a632b0c1b655990b4514d6f39fb71754a6c67c9772355`；
  来自A100 `reports/o378_roof_v35/after.sass`，全文件SHA256为`370dcb9c44ad10c5948f54c93b32c525524daf981f21fa645f6df5d770ca19a0`。

初次构建因include目录写成`csrc/include`失败，原日志保留在`reports/o378_roof_v53/`；
改为`include`后重新构建，所有有效结果来自`reports/o378_roof_v53_build2/`，没有复用失败产物。
NCU二进制留在A100同目录，仓库归档raw/source-SASS导出及启动日志，不提交`.so/.ncu-rep`。

```bash
python scripts/build_vector_conversion_probe.py --output reports/vector_conversion_recheck
python scripts/validate_vector_conversion_probe.py \
  --library reports/vector_conversion_recheck/libvector_conversion_probe.so \
  --output runs/vector_conversion_validation_recheck
python scripts/benchmark_vector_conversion_probe.py \
  --library reports/vector_conversion_recheck/libvector_conversion_probe.so \
  --output runs/vector_conversion_trace_recheck --samples 24 --rounds 3 \
  --warmup 50 --repeats 200 --inner 100
python -m unittest tests.unit.test_roof_v53_evidence
```

下一步仅将固定16元素转换接入内部GEMM59四模式接口，对照既有转换4，
重新测Cold/steady、缓存语义、payload/scale/MSE并审计旧GEMM机器码。
完成前不替换当前“完整流水线最佳”，不把上表两行相加冒充实测总时间。

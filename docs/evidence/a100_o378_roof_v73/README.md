# v73：O7/O8 行级转换与整数 metadata 融合

本轮只优化全K候选的在线准备；**GEMM使用同一个v67 CUfunction，正式默认、O3和5090不变**。
控制是v69的完整GPU准备，不是没有整数guard的旧转换5；因此不能跨run拼接累计收益。

## 唯一实现差异

|路径|权重在线准备|激活在线准备|
|---|---|---|
|v69控制|转换/组平方和 → 独立行metadata|转换/组平方和 → 独立行metadata → CTA guard|
|v73候选|同一CTA内完成转换、组平方和及行metadata|同一CTA内完成转换、组平方和及行metadata → CTA guard|

一个256线程CTA处理一行4096元素；8线程各转换16元素，覆盖一个G128，合计32组。
每组的平方和与scale code先放入256B shared memory，经过一次CTA barrier，由warp0
计算与v69相同的factor、anchor/base、加权范数及状态。其余转换、RNE、packing和有效scale表达式不变。

每次Cold少2个kernel启动，steady少1个；不再从global重读组平方和及scale code。
4096³下A/W平方和合计1MiB仍写出用于逐元素验证，**没有消除这块scratch或其写流量**。
逻辑读取减少不等于同样大小的实际DRAM流量减少，未用NCU定量归因。
INT32范围/FP32 epilogue保护、CTA一致回退、独立G128数学语义全部保留。

## 编译与对照约束

|源格式|转换/metadata寄存器/线程|Shared/CTA|Stack/local/spill|
|---|---:|---:|---|
|NVFP4|31|256B|0|
|MXFP8|32|256B|0|
|HiF4|31|256B|0|
|FP6实验变体|23|256B|0|

新4个入口各有1次CTA barrier、没有LDL/STL；15个旧准备入口编码SASS与v69逐条相同。
GEMM没有重编译，沿用原cubin及其同entry两路原生U4×S4/S4×S4、cp.async审计。
GEMM仍168regs、34304B shared、3CTA/SM，原有fallback静态spill不属于新增转换spill。

## 配对测试口径

A100，真实FP16 trace直接生成实验源格式；公共源格式量化不计时。
四样本初筛后扩大24样本；每次均warmup50、repeats200、转换inner100、单stream、预分配。
转换阶段批量摊销；GEMM/Cold/steady单次直接CUDA Event计时，初始执行顺序交错。
记录所有原始时间及CV失败，不筛最快轮。四样本与24样本分别报告，不能混算。

四样本初筛：转换配对吞吐O7/O8提高15.13%/10.77%，Cold提高1.24%/1.91%；
steady分别+0.46%/−0.05%，区间跨1，没有确认收益。全部64条输出/MSE与v67逐位相同。
初筛的`preparation_implementation`沿用了v69标签；原始数据保留，不回写篡改。
`candidate=1`及build/source/driver证据确定实际运行row-fused入口；后续commit
`4c581c2`仅修正计时元数据标签，CUDA库与计时方法未改，24样本记录使用正确标签。

## 24样本结果

一轮24样本×2后端×4模式×2路径，共384条记录。ms列为24个样本内median的median；
吞吐提升先对同样本配对再取median，故不等于两列总median直接相除。
置信区间为样本配对speedup的描述性bootstrap 95%区间，不代表独立模型/输入总体。

|后端|模式|v69控制 ms|v73候选 ms|配对吞吐变化|speedup 95% CI|
|---|---|---:|---:|---:|---|
|O7|Conversion-only|0.076370|0.066924|+14.23%|[1.137257,1.144959]|
|O7|Compute-only|0.477696|0.477184|+0.21%，无新GEMM优化|[1.000000,1.010707]|
|O7|Cold|0.567296|0.557056|+1.83%|[1.016483,1.020445]|
|O7|Steady-state|0.536576|0.528384|+1.21%|[1.003824,1.015504]|
|O8|Conversion-only|0.101629|0.091300|+11.15%|[1.107708,1.114430]|
|O8|Compute-only|0.481792|0.482304|−0.11%，无新GEMM优化|[0.993671,1.002146]|
|O8|Cold|0.591872|0.581632|+1.66%|[1.014060,1.021053]|
|O8|Steady-state|0.548352|0.545792|+1.13%|[1.001876,1.013133]|

|转换阶段|O7控制→候选 ms|O8控制→候选 ms|
|---|---|---|
|W|0.026086→0.022149|0.039493→0.033782|
|A（含CTA guard）|0.050243→0.044787|0.062252→0.057536|

转换总median减少约9.45/10.33µs；这是带guard准备路径的收益，不是更高Tensor Core峰值。
与没有全K metadata的转换5相比仍有额外成本，本轮没有重新配对59+转换5，
**不能把v69与v73不同run的百分比相乘，宣称新的累计实测**。

|后端/参考|输出MSE median|输出MSE mean|相对v69/v67|
|---|---:|---:|---|
|O7/O5|0.005536172273439|0.005053635851003|输出逐位相同，差值MSE=0|
|O8/O6|0.004411084910986|0.004381379299074|输出逐位相同，差值MSE=0|

384条结果metadata精确核对通过、finite FP32；仅O8的`layer_24_o_proj`保留12个CTA回退，
其余整数路径与原guard一致。不是删除安全检查或改变量化换取加速。

## 波动、安全性与结论边界

|后端/路径|选定指标CV≥3%数量：转换 / GEMM / Cold / Steady（各24）|任一阶段CV≥3%数量|
|---|---|---|
|O7控制|0 / 24 / 13 / 19|0 / 24 / 24 / 19|
|O7候选|0 / 24 / 20 / 23|0 / 24 / 24 / 23|
|O8控制|0 / 24 / 10 / 23|0 / 24 / 23 / 23|
|O8候选|0 / 24 / 15 / 21|0 / 24 / 24 / 21|

共享、未锁频GPU。单次GEMM后50次/前50次延迟median比值跨样本汇总约1.09–1.13，
Cold/steady约1.05–1.09，存在段内漂移，不只是个别离群点。
保留全部原始Event和GPU快照；不宣称严格CV验收，也不凭快照将每个离群归因于其他任务或频率。

**保留为全K路径的更优独立准备候选，不切换正式默认。** GEMM容量模型与约0.220347ms的必要MMA
下界未变；本轮没有缩小GEMM本身的差距，不将端到端改善冒充接近该下界的新证据。

GPU预检覆盖2后端×4模式×2路径×4模式化输入，共64项，加12项非法scale/范围/饱和等边界。
覆盖全零、随机、交替极值、宽scale、非默认stream；FP64语义参考rtol/atol=1e-3，
payload、有效scale、平方和、metadata逐元素核对，输出与v67逐位一致。
memcheck、synccheck各0 errors；racecheck 0 errors/0 warnings。
每种sanitizer均完成64+12项检查，**M/N≤128/256、K=4096，不是4096³的sanitizer覆盖**。
24样本4096³另行通过输出、MSE、metadata检查；本轮没有新增NCU。
33项本地单元/证据测试通过，重新从全部原始Event计算阶段统计、完整配对汇总，
并核对source SHA、15个控制入口机器码、新入口资源、计时/缓存标签及sanitizer日志。

## 复现与证据

实现commit `7cb5d7b`，元数据标签修正 `4c581c2`；本地实现、GitHub push后在A100 fetch/ff-only合入。
正式扩展SHA256仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
首批编译/初筛归档SHA256：`647c7807ae9cc1606054de5b72b71d69e04d4c4b637bc6e66e0ab367333f79f1`。
24样本/sanitizer归档SHA256：`dd405a48afbe008482da1a4e35716d12863f50b9796c01c5857161fc2bae9943`。

```bash
python scripts/probe_o78_row_fused_codegen.py --output reports/v73_rebuild
python scripts/benchmark_o78_row_fused.py \
  --gpu-build reports/v73_rebuild --output runs/v73_recheck \
  --samples 24 --rounds 1 --warmup 50 --repeats 200 --inner 100 --full-modes
```

输出目录须不存在。原v67 GEMM目录需保留；`--baseline`可显式指定。
`--validate-only`用于sanitizer；不用`--full-modes`时只测缓存GEMM，不能据此评估本轮准备收益。
完整SASS、build/resource日志、原始Event序列、源数据provenance和验证记录随本报告保存。

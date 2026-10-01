# v35：相同GEMM59的转换融合端到端对照（完成）

仅O7/O8的显式内部实验参数`conversion_impl`可选；默认0保持原转换和重排。
不改变正式默认、G128量化/scale、两路INT4 GEMM或5090。

|ID|转换路径|
|---|---|
|0|原自然布局转换 + G128-major重排，当前最佳GEMM59的对照|
|1|v25浮点融合转换|
|2|v34整数flat转换|
|3|v34整数目标布局转换|
|4|固定格式选择：MXFP8沿用1；NVFP4/HiF4/FP6采用3|

4的选择依据是v34隔离转换实验，不是逐样本autotune。
所有路径调用**同一个59**，由0/4同binary循环换序对比端到端。
转换每operand一kernel，直接产生GEMM布局；删除原中间payload的写回/重排。
仅为兼容诊断返回值，计时完成后导出自然布局，不在真正执行路径中使用。
全部source→fixed的解码、RNE、packing及scale生成成本仍在计时内。

计时沿用四模式、50次预热、200次Event、转换inner100。
Cold/steady直接测一次完整执行；转换only累计配对阶段样本后计算统计量，不把分段median相加。
保持原24个FP16 trace、paired O5/O6参考、逐位输出检查和FP64 MSE。

## 完整24样本结果

同binary、单轮、循环换序，50/200/inner100，共384条四模式记录，不删除离群。
以下变化为同样本配对吞吐变化，不是两列总体median的商。

|后端|模式|旧转换0 ms|候选4 ms|配对吞吐变化|描述性speedup 95%区间|
|---|---|---:|---:|---:|---|
|O7|Conversion-only|0.174162|**0.117701**|**+47.78%**|[1.47665,1.48044]|
|O7|Compute-only|0.516096|0.522240|−0.78%，无明确变化|[0.97945,1.02049]|
|O7|Cold|0.696832|**0.650240**|**+6.91%**|[1.04871,1.07729]|
|O7|Steady-state|0.619520|0.596992|+2.60%，未确认|[0.99667,1.04288]|
|O8|Conversion-only|0.151396|**0.128584**|**+18.18%**|[1.17464,1.18329]|
|O8|Compute-only|0.529664|0.528384|+0.81%，无明确变化|[0.99140,1.01741]|
|O8|Cold|0.674560|**0.656384**|**+3.15%**|[1.00754,1.05083]|
|O8|Steady-state|0.622336|**0.611840**|**+1.33%**|[1.00243,1.02260]|

确认本轮转换和cold改善；O8 steady有小幅收益，O7 steady区间跨1，不能确认。
GEMM机器码完全不变，compute差异不用于宣称新数学kernel更快/更慢。
GEMM主结果继续采用v33五轮测量，不把本轮单点与不同run拼接。

为什么转换收益没有全部变成端到端收益：转换只是部分开销；GEMM仍不变；
steady已经缓存W，因此NVFP4权重约2.32×的转换加速不直接改善steady。
各阶段来自不同计时上下文，不能用stage median相加构造total。

## MSE与稳定性

全部384条输出与旧转换+GEMM59逐位相同，优化引入的输出MSE=0。

|后端/参考|Median MSE|Mean MSE|
|---|---:|---:|
|O7/O5|0.005536172426666|0.005053635833762|
|O8/O6|0.004411084948645|0.004381379215302|

候选4的选定阶段CV≥3%数量，按conversion/compute/cold/steady（每项24）：
O7 `0/13/2/17`，O8 `0/13/8/18`。
检查任一阶段时，O7 `0/13/21/17`、O8 `0/13/19/19`；尤其不能隐藏cold中的GEMM阶段波动。
所有样本和离群均保留，单轮24样本也不等同于24次独立模型trace。
**这不是严格全阶段CV<3%的正式稳定性验收，正式默认不切换。**

## 正确性、审计与证据

- 288项GPU四模式检查、4项非法候选拒绝检查通过，含随机/零/交替及不同shape。
- 对原数学参考满足1e-3，所有融合方案与原59逐位一致；四模式缓存/计时metadata检查通过。
- 有限memcheck/synccheck各96项+4项拒绝检查，均0 errors，不代表全输入空间安全证明。
- 原生双INT4、异步复制指令审计通过。12个旧正式GEMM和148个旧候选编码SASS完全不变。
- 3个非目标mixed-binary函数codegen变化，完整mixed回归通过；全范围报告保留passed=false。
- 转换审计8个实例零stack/local，新目标布局无F2I；当前GEMM59原有spill未消失。
- NVFP4旧查表有8B stack、4条静态LDL/STL；新寄存器LUT无该访问。
  `nv4_lookup_analysis.json`记录静态对照，不把静态条数当动态流量或完整性能归因。
- 本轮未重跑NCU：目标GEMM的SASS完全不变，转换隔离实验及直接端到端测量用于验证本次改动。

实现c2c3efec1c050c36763add8166efc78253db79c8。
binary SHA-256：`273b846dd55ebca0028f556c60d653a49d35b7756685bb8707c05e30ae2270af`。
35个文本证据经SHA/成员路径检查后提取；归档SHA-256：
`e3dd5c08c611e554d35ddb247db9be8a7c210cfd03f2aa0b96c21a7ad7b096fc`。

结果与每次Event：`runs/o378_roof_v35_four24/{results.jsonl,summary.json,environment.json}`。
审计、codegen、预检、回归、安全检查和单样本smoke全部保留。
`tests/unit/test_roof_v35_evidence.py`从原始样本重新计算median/配对区间，核对MSE及计时口径。

```bash
python scripts/validate_conversion_pipeline.py --output runs/conversion_pipeline_validation
python scripts/benchmark_conversion_pipeline.py --output runs/conversion_pipeline_four24
```

下一步优先将已验证的LUT/目标布局思路迁移至O3转换，再考虑向量化读写；
纯GEMM仍需另行减少发射等待或数学后处理，不能把转换加速冒充其容量上界改善。

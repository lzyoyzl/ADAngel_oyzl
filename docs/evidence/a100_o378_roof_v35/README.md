# v35：相同GEMM59的转换融合端到端对照（待验收）

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

需重新编译、四模式小形状验证、有限sanitizer、SASS/codegen回归、24样本完整性能/MSE后，
才能宣布端到端收益。当前文档不代表该验收已经完成。

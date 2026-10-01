# v49：N64 权重片段读取与计算重叠（待测）

目的：针对 shared→寄存器供数后的等待测试局部 lookahead。不是重新扩大 CTA、
warp 数或 N partial 窗口，也不是跨 G128 合并。正式默认和 5090 不改。

|方案|B fragment 读取时间|每路 partial/线程|作用|
|---|---|---:|---|
|0|原54/59按 N64 片段读取|16|同一SASS对照|
|1|当前 N64 开始计算前，额外读取下一 N64|16|较长供数重叠窗口|
|2|完成当前片段第一个 M atom 后，读取下一 N64|16|缩短额外寄存器活跃时间|

两候选只增加下一片段的 B 寄存器缓冲，不增加 shared 空间、global/shared 总读取次数，
不改变原始四 N atom 的 INT32 部分和与 FP32 group 顺序。
下一片段复用已读寄存器，而非重复读取 shared。边界为编译期，最后一片不预取。
v48 N128 同时扩展 B 和 partial，本轮只保留 B lookahead，因此不是重复 v48。
编译器也可能已经提前调度，必须通过实际 SASS/NCU 判断而非按源码推断重叠。

保持 CTA64×128×128、2×2 warp、128线程、原 cp.async.cg、G128-major、
O3三stage/O78两stage、scale语义、两路原生U4/S4和S4/S4、FP32输出。
风险：额外 B 寄存器挤压地址与临时结果，引起spill或更差发射。

验证：原54/59控制编码SASS完全相同；实际CuTe B坐标复用v48检查；
180项有限预检、memcheck/synccheck/racecheck；24真实样本×3轮、50/200配对Event，
逐位对照54/59并重新计算O3/O0、O7/O5、O8/O6的FP64 MSE。保留全部CV失败。
若初筛有收益再做独立确认及四模式；否则不替换最佳、不重测到通过。
必要时NCU复核LDSM/MMA/I2F/FFMA次数、local访问、eligible warp与具体等待点。

理想重叠容量下界仍由原始数学工作约束；它不是可达性能承诺。
跨G128整数对齐仍待明确确认，本轮不实施；magic实验保持停止。

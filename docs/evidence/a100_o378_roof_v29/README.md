# v29：O3 行 scale 移至 epilogue（候选51/52，待 A100 验收）

保持最佳41/42的CTA64×128×128、4 warp、G128-major输入、B fragment复用，
以及两/三阶段cp.async。仍使用两路原生INT4、每个G128自己的W scale、FP32跨组累加及输出。
O7/O8的A scale随G128变化，不能直接使用本优化，入口明确拒绝它们。

O3旧数学式：`sum_g(P_g * (A_scale[row] * W_scale[col,g]))`。
候选：`A_scale[row] * sum_g(P_g * W_scale[col,g])`。
G128顺序和每组INT32 partial不变，原UE8M0 scale没有合并或重新量化。
这在实数中相等，但改变FP32舍入，所以需要明确`--allow-reassociation`，
检查FP64语义参考、相对原实现误差，以及24样本对O0的MSE回归。

本地独立CUDA12.5编译通过，两个实例均168寄存器；51为8B stack、52为16B stack。
162项CPU契约/既有证据回归通过。这不是A100性能结果或零spill结论。

目标是减少循环内的scale组合和行scale寄存器活跃区间；**本候选没有减少I2F次数**。
旧O3已通过指数位运算合并scale，因此不能夸大为每组省掉一次浮点乘法。
候选每输出最后新增一次FP32乘法；实际调度、资源与性能必须测量。

新中间值尚未乘A_scale，可能比旧值大。两个host入口都显式检查
`max(W_scale) * K * 128 * 8 <= FP32_MAX`和最终结果保守上界，
并拒绝UE8M0 code0的subnormal W scale及非法code255；不满足时明确报错，
不偷偷改用另一个算法。普通正式后端仍保留原输入范围。

独立device header/TU；正式默认和RTX5090不变。
流程：本地编译/CPU契约→GitHub→A100同步构建→代码保持/INT4审计→GPU数值/安全检查→
4096初筛→真实24样本MSE/配对性能；只有有收益才进入多轮和完整四模式验收。

另一个“整数指数对齐后跨组INT32累加”方向尚未实现，等待用户确认累加类型是否允许改变。
它与本候选不同，不能把检查指数跨度当作该方向的数值或性能验收。

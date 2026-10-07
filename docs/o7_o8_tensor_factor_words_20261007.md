# O7/O8：系数 MMA 的直接寄存器输入（v130）

这是对[v129编译失败](o7_o8_tensor_factor_20261007.md)的**一次具体packing修正**，不是换layout或扫参数。
v129的54条PRMT说明：构造uint8 fragment再交给MMA，会引入额外byte拼接。
v130保留相同CuTe坐标和rank-one U8×U8 MMA，直接调用固定CUTLASS的U32寄存器接口。
每个A寄存器/B寄存器只有最低byte可能非零；其余byte确为零后才可这样构造。
Host CuTe检查全部128线程、A/B坐标和30720个byte，对照v129原fragment逐字节验证，禁止猜lane规则。

不改变两路INT4主点积、G128量化/scale、全K整数顺序、原溢出guard、新增U8范围guard与fallback。
旧v129失败记录保留，不将原失败改写为成功。正式默认与5090不变。

投入门槛**不放宽**：allocated≤168、热local≤1、静态循环≤383×1.01、IMAD族减少≥20%；
同entry64条原生INT4+16条系数INT8、16 LDSM/10 copy/1 barrier、旧控制完整编码不变。
失败即停止，不继续枚举packing/布局；通过后仍须检查双侧factor真实覆盖≥50%、实际资源、
GPU数值/MSE/安全，再进行完整24样本三轮配对。额外range guard计入实际GEMM，不免费。

当前状态：本地实现与CPU检查阶段，没有候选GPU性能或MSE结果。

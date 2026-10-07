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

## A100 编译结果：packing 修正有效，候选仍停止

源码先推送 GitHub，再在 A100 fetch/ff-only 到 `6e504c82ba9316a0a7ab567f4eb1fd707457c6ec`。
CUDA 12.8.93 / pinned CUTLASS。原 v78 对照完整机器码不变；
CuTe host 验证 8192 个输出坐标和 30720 个输入 byte，直接 U32 与原 uint8 fragment 一致。

| 同一整数热循环 | 原 v78 | byte 构造 v129 | 直接 U32 v130 |
|---|---:|---:|---:|
| 静态指令 / G128 | 383 | 461 | 406 |
| IMAD 族（含地址等） | 158 | 115 | 102 |
| 普通 IMAD | 128 | 69 | 68 |
| PRMT 字节重排 | 0 | 54 | 0 |
| 热 local 读 / 写 | 0 / 0 | 3 / 3 | 3 / 3 |
| Allocated registers / thread | 168 | 168 | 168 |
| 活跃寄存器峰值 | 166 | 166 | 166 |
| 两路原生 INT4 MMA 合计 | 64 | 64 | 64 |
| 系数 U8×U8 MMA | 0 | 16 | 16 |

**消除了构造 byte fragment 的冗余，但没有让总成本低于旧最佳。**
相对 v129，静态指令减少55条（11.93%）；相对 v78，IMAD族减少35.44%，
但总静态循环仍多6.01%，而且有3条LDL.LU、3条STL。
其他额外工作仍在：系数供数/选择、16条额外MMA及控制，
例如 LOP3 6→30、CS2R 0→10、额外 BSSY/BSYNC/WARPSYNC。
这不是新发现的 HBM 带宽瓶颈，而是额外计算、供数与寄存器工作的取舍。

同 entry 的原生 S4×S4/U4×S4、U8系数MMA、16 LDSM/10异步copy/1 CTA barrier均通过；
原工作量≤1.01倍与热local≤1两个门槛仍失败。
ptxas为整个新entry报告32 B stack、76 B spill stores/64 B spill loads；
这些字节数不是每组动态流量，也不能用资源表LOCAL:0否认SASS中的local访存。

**停止这个精确U32修正版，关闭此系数外积路线，不放宽门槛、不追加packing/布局扫描。**
范围guard的额外启动读取还没有算入上述热循环对比，因此不能将它视为免费。
未运行候选GPU，没有新Event、MSE、CV、sanitizer、NCU或实际occupancy结果；
静态指令减少11.93%不是实测加速，静态多6.01%也不是实测慢6.01%。
不追加完整24样本性能测试，当前最佳和正式默认/转换/5090保持。

本地及A100的7项代数/source/既有证据回放测试通过（不是GPU数值验收）。
原始PTX/SASS、日志、host坐标、liveness及SHA见[冻结证据](evidence/a100_o378_roof_v130/README.md)。
正式扩展SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

## 对后续优化的约束

减少标量系数乘法不能孤立评估：即便修好了输入packing，
新增Tensor服务、fragment供数和寄存器生命期仍可能使路线不合算。
后续不重复此路线或已否定的相邻参数；新方案必须减少主路径实际工作或改变有证据支持的依赖。
跨平台能复用“避免拆开再拼回”的原则；寄存器接口、lane坐标与原生整数能力必须重新验证，
不声称该淘汰候选已有跨平台收益。

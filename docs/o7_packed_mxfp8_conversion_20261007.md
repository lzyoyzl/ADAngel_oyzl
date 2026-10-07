# O7：packed MXFP8 转 Q8（v126）

状态：**A100 编译审计完成，未达到预设投入门槛，停止该候选。** 没有运行候选 GPU kernel，不改正式默认、当前最佳或5090。GEMM目标仍未达到。

## 动机与查重

v125缩短MMA链反而增加整数工作；不重做链数、CTA、pipeline或shared因子缓存扫描。
另一个设想是把IMAD换成DP2A以转移执行管线，已取消：NVIDIA说明GA100的IMAD、IDP均使用FMA管线，不能声称换指令就能卸载该管线。[NVIDIA开发者说明](https://forums.developer.nvidia.com/t/separate-cuda-core-pipeline-for-fp16-and-fp32/302018/7)

本轮采用有正面先例的packed转换：v123在O8有效，但其E2M3/F=2解码不能直接用于O7的E4M3/F=−2。
不同于v34标量取整、v106逐元素warp查表、v118权重nibble布尔解码，不重测原方案。

## 唯一新机制

一个32位字并行处理4个E4M3编码，根据指数位选择精确的RNE除法或整数左移；以byte隔离的符号处理生成Q8，再用DP4A计算精确平方和。无需逐元素浮点转换、取整或warp查表。
所有254个合法编码、正负零、ties-to-even、byte carry、packing均核对。两个NaN编码仍由既有源格式契约拒绝；内部算术保留旧decoder的数值行为并不意味着接受NaN源数据。

对照使用已测v106激活转换和v118权重转换，候选只替换激活decoder。两边都保留v73行融合metadata/guard、同一v78原生双INT4 GEMM、量化、FP32输出及四种计时口径。

## 预设投入门槛

原LUT激活和packed权重entry编码不变；相对LUT转换至少减少5%静态指令；不超过32寄存器、无local/stack，shared仍256B、1次CTA barrier、4条DP4A，实际仍8CTA/SM。
门槛通过后才进行GPU数值和完整24样本×三轮四模式配对，warmup1000/repeats200/inner100；不做小规模性能初筛。失败即停止，不扫描相近Boolean/LUT参数。

编译指令减少不是性能结果；只有直接Event测试可确认conversion、Cold和steady收益，也不能把转换收益宣称为GEMM吞吐提升。

## 实际编译结果与决定

| 指标 | 已测 v106 LUT 激活转换 | 新 packed 候选 |
|---|---:|---:|
| 整个转换 entry 静态指令 | 456 | 464（+1.75%） |
| 寄存器/线程 | 31 | 31 |
| CUDA 查询的 CTA/SM，256线程/CTA | 8 | 8 |
| shared / stack / local，bytes | 256 / 0 / 0 | 256 / 0 / 0 |
| CTA barrier | 1 | 1 |
| SHFL.IDX / LOP3.LUT | 20 / 95 | 4 / 122 |
| 原生 signed DP4A | 0 | 4 |

旧 LUT 激活 entry 与 v106 的完整 SASS 编码相同，旧 packed 权重 entry 与 v118 的完整编码相同。
新转换减少16条查表 shuffle、加入4条DP4A，但E4M3/F=−2的多种RNE除法、指数选择和byte隔离需要更多整数逻辑：LOP3 95→122、IADD3 10→21。整体没有减少指令，也未改善驻留数。

**按预设门槛停止，不做GPU数值/性能/NCU或相邻Boolean/LUT扫描。** 静态指令增加1.75%不等于实测慢1.75%；未测也不能写成0%提升。CPU编码、RNE、packing和平方和检查通过，但不是GPU正确性或MSE验收。当前历史MSE和最佳性能保留，本轮没有新成绩。

首次源码 `058521dbb71ec12006de964943d0a64354a13696` 已成功编译，但审计将Mx8误写为enum编号2，无法定位entry。真实定义为`Nv4=0, Mx8=1, Hif4=2, Nv6=3`。修正及真实SASS定位回归的commit为 `436e974e355d3c3e85eea32603db925546ff8109`，未修改CUDA算法或投入门槛；两次编译和首个失败日志均保留。所有源码先推GitHub，再在A100项目内fetch/ff-only。最终独立库SHA为 `906dde016ff9516edda4f1ee682932dd5816f0706c66c698ea1e0c846cc837fe`。

[冻结原始证据与重放说明](evidence/a100_o378_roof_v126/README.md)记录完整SASS、资源、生成代码、来源commit及SHA。正式扩展SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

## 对瓶颈和移植的启示

这一轮是转换阶段的投入筛查，不能代替GEMM瓶颈验证。当前GEMM仍由两路INT4、整数factor加权、寄存器依赖和供数/同步共同限制；已有逐组FP32转换被移到尾部，不能再据旧版本诊断优化。详见[当前瓶颈与平台迁移](o3_o7_o8_bottleneck_portability_20261007.md)。

**可以迁移方法，不能迁移收益结论。** 相同A100上，O8的E2M3/F=2 packed转换有已测端到端收益；O7的E4M3/F=−2却没有通过指令门槛。换格式已需重新评估，换GPU更应重验解码、dot指令、fragment布局、寄存器/驻留及直接端到端时间。量化语义、精确整数对齐与guard、融合转换和寄存器partial的原则可复用；A100的原生INT4 SASS、cp.async参数及加速比例不可直接复制到5090或其他平台。本轮未进行跨平台测试。

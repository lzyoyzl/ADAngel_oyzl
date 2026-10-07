# O3：跨 G128 的 A fragment 加载与整数收尾重叠（v128）

## 假设与边界

对象为 A100 最佳独立 O3 v89，不切换正式默认，不修改 O7/O8、5090、转换或量化。
当前最后一个 N64 的两路 INT4 MMA 完成后，旧 A fragment 已经失效，但还有整数 factor 加权。
候选先加载下一 G128 的 A 到**同一组寄存器**，再完成本组末尾加权，尝试隐藏 LDSM 延迟。
不增加 A 双缓冲，不增加 MMA、payload/factor 字节或最终 store。

查重：v49/v96/v97 是跨 N 的 B/partial 排布；v42 是整批 global 预取时机；v127 是全 K Af shared 面板。
本轮改变的是**跨 K 的 stage 交接和 A shared→register 加载位置**，不重复上述候选或扫描 tile/stage。

仍为 64×128×128 CTA、128线程、3-stage/50688 B shared、32 partial 寄存器、M8 CTA 遍历。
先缓存旧 group 的 W factor，再执行 stage wait/barrier，避免另一 warp 覆盖旧 stage 时仍读取系数。
32组总计仍为32次 barrier：1次 prologue，31次组间交接。末组不加载不存在的下一组。
整数乘加顺序、host/device guard、fallback、最终 FP32 缩放和输出完全保留。

## 先固定的投入门槛

1. 编译审计：原控制完整机器码不变；同一候选 entry 使用原生 U4×S4 与 S4×S4、cp.async.cg。
2. 同一热循环32/32条原生 MMA、16 LDSM、9 async copy、1 barrier；实际 loop 外 prologue也核查。
3. SASS中8次下一组 A 的 LDSM确实在最后一条 MMA之后，且后面至少有8次已识别的旧 partial 加权。
4. 静态循环指令最多增5%、allocated registers≤168、热 local至多1条；小spill仅因用户已允许，不据此放宽正确性。
5. 通过后检查实际3CTA/SM、old-factor读取与barrier位置、synthetic/edge/MSE及有限mem/sync/race检查。
6. 候选若值得进入性能验收，直接24真实样本×3轮，warmup1000/repeats200，同进程输入/stream交错配对。
   不做小规模性能初筛；正向再补四模式。负向停止，不扫描相邻调度或移植其他case。

CPU source/ring测试仅验证计划结构，不是GPU数值或并发安全证明。静态顺序不等于真实硬件重叠。

## A100 编译结果：门槛未通过，停止

源码 `67eff20593662f4c98b1d4ab549921ba67123c9d` 已先推送 GitHub，再同步 A100 编译。
CUDA 12.8.93、固定 CUTLASS；原 v89 对照完整机器码一致，没有重编译正式扩展。

| 同一整数热循环的编译指标 | 原 v89 | v128 |
|---|---:|---:|
| 静态指令 | 323 | 374（+15.79%） |
| allocated registers / thread | 168 | 168 |
| 静态活跃寄存器峰值 | 166 | 153 |
| 热循环 local load / store | 0 / 0 | 0 / 0 |
| 原生 signed / unsigned INT4 MMA | 32 / 32 | 32 / 32 |
| LDSM / async copy / CTA barrier | 16 / 9 / 1 | 16 / 9 / 1 |
| 普通 IMAD | 65 | 67 |
| LOP3.LUT / S2R | 4 / 2 | 21 / 8 |

**目标调度确实生成了：** 候选 SASS `0x60e0..0x6150` 有8条下一组 A 的 LDSM，
之后从 `0x6170` 起有32条使用旧 partial 的加权 IMAD。不是仅修改了源码、机器码未变化。
同一候选 entry 的原生 U4×S4/S4×S4、cp.async.cg 审计通过。

但新旧 stage 分离后增加了位操作、线程索引获取、地址与控制指令，
例如 LOP3 +17、S2R +6、SHF.L +6、SHF.R.U32.HI +3；静态循环总计增加51条。
主要 MMA 和加权没有减少，活跃寄存器虽然下降，**实际分配仍为168**。
这是“改善依赖排布，但增加供数/控制工作”的取舍，不是已经测出 stall 降低。

超过预设静态指令最多增加5%的投入门槛，**不运行候选 GPU、不做相邻排布扫描，也不扩展 O7/O8**。
没有新 Event 延迟、MSE、CV、GPU sanitizer、NCU 或实际 occupancy 数据。
**不能写成“性能下降15.79%”，也不能写成“MSE已验证不变”。**
本轮只是编译审计淘汰，旧最佳 O3 v89、O7/O8 v78及已有转换候选保持。

原始 PTX/SASS、liveness、编译日志和哈希见 [15份冻结证据](evidence/a100_o378_roof_v128/README.md)。
本地与A100 CPU source/ring测试均通过；正式扩展 SHA 保持
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

实现：[代码生成与门槛](../scripts/probe_o3_crossk_load_codegen.py)、[候选入口](../csrc/sm80/roof_o3_crossk_load_probe.cu)。

## 可移植性

可迁移的是“复用失效的寄存器、把下一块加载放到独立收尾计算之前”的软件流水线思想。
CUDA/CuTe fragment、共享内存生命周期、barrier语义、MMA延迟和资源预算必须按平台重做；
不能声称此编译门槛淘汰的候选已经改善A100，或把A100的原生INT4速度搬到5090。
完整瓶颈证据、当前最佳数字及平台边界见 [瓶颈与迁移报告](o3_o7_o8_bottleneck_portability_20261007.md)。

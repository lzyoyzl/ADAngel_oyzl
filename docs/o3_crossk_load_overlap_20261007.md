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

## 状态

已实现独立生成器和包装入口，CPU契约2项通过；等待A100编译门槛。
当前没有新增性能、MSE、GPU安全或跨平台结果，旧最佳保持。

实现：[代码生成与门槛](../scripts/probe_o3_crossk_load_codegen.py)、[候选入口](../csrc/sm80/roof_o3_crossk_load_probe.cu)。

## 可移植性

可迁移的是“复用失效的寄存器、把下一块加载放到独立收尾计算之前”的软件流水线思想。
CUDA/CuTe fragment、共享内存生命周期、barrier语义、MMA延迟和资源预算必须按平台重做；
不能声称此未测候选已经改善A100，或把A100的原生INT4速度搬到5090。

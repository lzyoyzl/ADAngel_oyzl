# O7/O8 v120：固定 CTA 异构调度候选

## 查重与唯一改动

v63 是旧逐组 FP32 路径的统一四链，v78 是所有 CTA 的统一八链；v92、v96/v97 改统一链数/前视。
v41 分离 producer，v89 改输出 tile 顺序，v104 增 warp，v119 改 asm 编译依赖。
本候选不重跑以上方案：在当前 full-K 精确整数路径中，固定 **1/3 CTA 四链、2/3 CTA 八链**。
分派位于整个 K 循环之外，对 CTA 全体线程一致；不扫描比例、种子或邻近参数。

动机来自当前 O8 预热 NCU：issue 46.90%、eligible 0.7464，wait/math 是主要未发射 PC 样本。
尝试让不同 CTA 的 MMA 与整数加权阶段错开，但不能假定相邻三个 CTA 同驻一个 SM，
也不能把静态四/八链数当作硬件并行度；需要最终 CUDA Event 配对测量证明收益。

## 保持不变与预设门槛

双原生 INT4、CuTe 映射、源量化/G128 scale/full-K 安全 guard、v73 准备、最终 FP32 和所有默认不改。
CTA 仍 64×128×128、128线程、两阶段、34304B shared；每输出整数 group 顺序不改。
四链只改变不同输出之间的发射顺序，不改变每个输出的精确数学。

编译 gate：旧控制完整 SASS 相同；两种整数路径真实存在；各自 64 MMA、16 LDSM、10 copy、
原 barrier 数量；零热 local、最多168regs；加权静态工作不超过原383条的1.03倍。
这是 phase-overlap 潜力 gate，不要求指令总数减少，也不是速度或 GPU 正确性验收。

通过后要求实际 CUDA Driver ≥3 CTA/SM，先做 GPU 数值/边界校验，再直接进行全24样本三轮
配对，每轮warmup1000、repeats200；不做小规模性能初筛。逐位输出、MSE、metadata 与 v67 回归。
负向则停止此机制，不追加 Cold/conversion/NCU；正向才扩展四模式和安全检查。

## 本轮状态

已实现独立候选与固定编译 gate，等待 A100 编译结果。
尚无新的 GPU 数值/MSE、Event 性能或安全结论；当前最佳及正式默认不变。

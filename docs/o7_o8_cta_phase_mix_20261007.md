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

## A100 编译结果：停止，不进行性能测试

运行源码 commit：`9d9b8a103caff66318da3e2f63c3023c92bb97f8`，先本地推送，后项目内 fetch/ff-only。
旧控制完整 SASS 编码不变，同一候选 entry 两种整数循环及原 fallback 均存在。
各整数路径仍有32条S4×S4、32条U4×S4、16条LDSM、10条cg copy和1条barrier。

|编译路径|循环静态指令|静态未完成MMA链峰值|每G128热local读取|Allocated regs|
|---|---:|---:|---:|---:|
|原v78控制|383|8|0|168|
|混合候选路径A|368|8|4条LDL.64|168|
|混合候选路径B|367|6|3条LDL.64|168|

候选整个entry为32B stack、72B spill load/store；这些编译器字节数不是动态访存量。
源码四链路径被编译器跨输出交错成静态六链峰值，不是预设的四/八链对照；峰值也不代表硬件并行度。
循环指令稍少不能抵消或定量评估新增local读取和依赖；未通过零热local及预设路径门槛，
加权工作字段因此为null，不能据此声称实测慢或快。

停止候选，不放宽gate、不扫描比例/种子/邻近链数、不迁移O3。此前允许少量spill，
不意味着所有spill候选都值得测试；本候选还未保持预设调度结构，未进入GPU验收。
无新Event速度/MSE、conversion/端到端、NCU或sanitizer结论；不能将“未测试”报告成0%提升。
本地与A100的15项CPU/证据回放测试通过，属于源码/编译证据核对，非GPU正确性证明。
当前最佳、正式默认/扩展及5090不变，GEMM目标尚未达到。

原始PTX/SASS/liveness/日志及SHA见[完整编译证据](evidence/a100_o378_roof_v120/README.md)。

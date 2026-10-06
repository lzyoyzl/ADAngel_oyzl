# O7/O8：独立 warp 流水线候选（v109）

结论：**编译及原生指令审计完成，潜力门槛未通过，停止候选；最佳版本不变。**
没有运行候选 GPU kernel，不提供新的延迟、MSE或内存安全结论。

## 为什么不是重复优化

本次让四个 warp 分别拥有私有的32×64输出与shared输入，在最后一段输入加载后复用同一个buffer，
尝试避免每个G128等待同CTA其他warp。
它不是已测的 v41 producer warp、v95共用buffer的mbarrier槽位握手、v98八warp几何、
v107驻留轴交换或已有stage/tile/cache/链数扫描。

仅一个固定候选：CTA64×128×128、128 threads、八MMA链、32个寄存器partial。
四个私有单buffer合计34304B，与原两stage的总shared容量相同。
代价是CTA内不再跨warp共用输入，逻辑payload读取量翻倍，thread/output ownership也发生必要变化。
这不是严格单因素微基准，也不是免费删除同步。

G128顺序、两路U4/S4与S4/S4、factor数学、全K整数guard和最终FP32输出不改。
fallback源码保留原方案；正式扩展、production默认、转换和5090均不改。

## A100编译结果

|指标|当前v78|v109候选|
|---|---:|---:|
|整数循环静态指令|383|433（+13.05%）|
|每线程分配 / 活跃峰值寄存器|168 / 166|168 / 164|
|原生INT4 MMA / LDSM指令|64 / 16|64 / 16|
|循环异步搬运指令|10|18|
|循环local-memory指令|0|8|
|候选entry stack / spill store / spill load B|0 / 0 / 0|32 / 44 / 36|

这些是二进制工作量和资源数据，**不能换算成实测变慢13.05%**；local指令计数不是字节流量。
三CTA寄存器预算没有变好，又增加copy/地址/metadata工作和热spill，未通过预先定义的无热local门槛。
因此不进行24样本性能/MSE/四模式/NCU测试，不为此方向继续相邻buffer变体，也不迁移到O3。

## 同步与正确性证据的边界

Host-only CuTe校验覆盖8192个输出的唯一owner、6144个byte/nibble布局检查，全部通过。
同一候选entry的PTX和SASS均确认两路原生INT4及cg异步copy，旧v78控制entry的机器码逐字相同。

候选整数循环没有CTA `BAR.SYNC*`；PTX包含三个 `bar.warp.sync`，SASS没有显式WARPSYNC。
**不能仅据SASS缺少该opcode就声称没有同步，或据PTX/坐标检查就声称GPU内存安全已通过。**
编译器可以改变指令实现，实际重用安全仍需GPU正确性与race/synccheck验证；本候选因工作量门槛已停止，未做该验证。
源码采用warp同步的依据是[NVIDIA CUDA同步定义](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#synchronization-functions)；
候选代码只保留为内部研究记录，不进入公开后端。

首次编译的CPU gate仅精确匹配`BAR.SYNC`，归档后已补齐`BAR.SYNC.DEFER_BLOCKING`等前缀识别及测试。
这只是解析修正，没有重编译或更改cubin；对本次原始gate结果无影响，仍然失败。

## 证据与当前结果

- 实现/编译提交：`8f7ac987db010819af05cfff22bb4c81ebc9dce8`，先推送GitHub再同步A100。
- [原始编译、坐标及指令报告](evidence/a100_o378_roof_v109/reports/o378_roof_v109_codegen/codegen.json)。
- 同目录保留PTX、SASS、liveness、resource及全部build日志；编译只执行一次。
- 完整归档SHA256：`4a5788356b1267e5324c843ed77f184fa52dd8b7277c4ffe2d738c26ba1c3e70`。
- 原生扩展前后SHA：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，相同。
- 3项CPU机制测试及4项冻结证据重放测试通过；连同v105/v108回归，本地21项通过。没有新GPU性能或MSE结果。

保留原有O3 v89、O7/O8 v78+v73主基准。目标尚未达到；本轮收益是排除一种同步解耦方向，
不是性能提升。即使去掉CTA等待，增加的供数/地址/寄存器开销也必须被计入，不能只数减少了多少barrier。

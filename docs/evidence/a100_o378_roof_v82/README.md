# v82：相同三 CTA 驻留上限下的原生 INT4 发射容量诊断

**三 CTA/SM 本身并不排除高 INT4 吞吐。真实最佳仍为 O3 v79、O7/O8 v78+v73；正式默认不变。**
本轮是容量诊断，不是改过量化的 GEMM，也不产生新的原实验 MSE 或端到端成绩。

## 测试方法

A100 108 SM，CUDA12.8；2048 CTA、128线程/CTA。用50688B动态shared padding限制最多3 CTA/SM，
循环256组，每warp每组64条原生m16n8k64 INT4 MMA。每组两段、每段8条逻辑partial链。
输入片段只在入口读取；主循环没有payload加载、scale处理或64个真实最终输出accumulator。
所有组/链进入checksum，避免工作被删除；含少量checksum与入口/出口工作。
负数、极值等4种packed pattern、2种循环长度、3条路径共24项checksum检查通过。

50预热，200次循环换序、CUDA Event单次测量；分配在计时外，保留全部600个样本。
这不是运行原4096³矩阵，只是匹配其CTA数量和每G128的MMA工作。

|路径|256组median ms|CV %|物理INT4 TOPS|按32组工作量折算ms|
|---|---:|---:|---:|---:|
|仅S4×S4|1.792000|0.433|1227.13|0.224000|
|仅U4×S4|1.792000|0.434|1227.13|0.224000|
|high→high→×16→low→low合并链|2.022400|0.455|1087.33|0.252800|

物理运算数为 `2048 × 4 warps × 256 × 64 MMA × (2×16×8×64) = 2,199,023,255,552`。
最后一列仅将256组时间除以8，**不是实际4096³ GEMM延迟、有效吞吐上界或相对最佳的加速比**。
同质路径的跨段调度没有被限制为与合并路径完全相同，不能将差值全部归因于signedness切换。
合并路径源码每段8条独立链，但实际SASS最多5条链已开始未结束；真实v78为8条。
这只是程序次序上的存活统计，不是硬件在途数；诊断并未复制真实kernel的完整调度。
两条单类型路径66regs、无spill；合并路径40regs，8B指针spill位于循环外。三条均无主循环内存指令。
真实kernel是168regs/线程，不能据此说其寄存器依赖和访存成本也被复制了。

## NCU交叉验证

合并路径50次预热后采集1次：`--set full --cache-control all --clock-control none`，50 passes。
精确symbol `adangel_capacity_merged`，静态SASS与source导出逐opcode指纹一致。

|项目|结果|
|---|---:|
|NCU Duration ms|2.019712|
|SM clock GHz|1.409696|
|动态warp MMA|134,217,728|
|硬件INT4运算数|2,199,023,255,552|
|硬件INT8运算数|0|
|Tensor active %|87.297357|
|Issue active %|26.711009|
|Eligible warps / scheduler|0.502801|

普通Event统计与NCU分开。NCU日志里的`raw_ms=4086.51978`包含profiling replay开销，不能作为kernel延迟。
NCU硬件运算数与源代码模型一致，源指令动态总数327,401,472也与raw计数一致。

## 对后续优化的影响

- 不再把“只有3 CTA”或“issue active不足50%”单独当作主要性能缺陷：本诊断仍能充分使用Tensor管线。
- 真正kernel与容量诊断之间还多了payload供数、组scale、完整accumulator与它们的依赖，不能用两者时间相减分配开销。
- 当前真实v78 SASS已将下一N64的LDSM/部分MMA穿插在上一片段的后处理中；“手动重叠”不是尚未存在的技术。
- 后续只测试能明确减少实际工作、且不增加大量重复fragment读取的结构性改动；不重复强压寄存器、八warp和metadata-copy扫描。

原必要MMA理想容量下界约0.220347ms @1410MHz保持不变，目标尚未达到。
本轮没有新的conversion、Cold、steady、原实验MSE或真实kernel加速百分比。

## 证据与复现

初次审计把循环外8B指针spill也视为失败，未运行GPU计时；修正为明确检查主循环无访存，并保留冷路径spill记录。
两个编译目录均保留；未改变device数学来绕过审计。源实现219d4d8，修正审计f14a8b5。
完整归档（含未提交Git的可执行文件与ncu-rep）SHA256：
`84565d4b3eaab961f2a879b29d2a6d3d3b1b81eb3eaab3f22d158702c81a1c7d`。

```bash
python scripts/probe_a100_mma_issue_capacity.py --output reports/capacity_recheck --run
python scripts/analyze_mma_issue_capacity.py --input reports/o378_roof_v82_capacity_checked
python -m pytest tests/unit/test_mma_issue_capacity.py tests/unit/test_roof_v82_evidence.py -q
```

输出目录须新建。原始文本、600次Event和NCU CSV在本目录reports中；不把没有运行的真实MSE/四模式测试写成通过。

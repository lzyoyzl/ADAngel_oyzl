# O3：跨 K 加载候选的寻址修复（v134，编译门槛停止）

仅保留两路原生 INT4；不测试单路INT8，不改正式默认或5090。

v128已将下一组A fragment加载提前到当前组32次整数加权之前，但热循环323→374：新增线程坐标、位操作和地址构造，未过原投入门槛，尚未运行候选GPU。
本轮不是重跑已确认负收益的方案，而是对这项明确的额外工作做一次修复。

从原CuTe copy partition取得4个stage0的shared源地址，GEMM入口计算后保留在寄存器。一次同lane shuffle保持地址值完全不变，但使其对前端优化器不透明，尝试避免循环内重新展开坐标；该额外指令也计入GEMM。
组内只加stage偏移与高位平面偏移，仍使用原公开 `ldmatrix.x4`，不修改片段映射。先在CPU用真实CuTe遍历128线程、3stage、2平面、2个K64及2个M片段，验证全部3072个地址和12288个destination word。

首次CPU编译暴露Copy partition的嵌套shape与预期不同，断言阻止后续GPU编译/执行；保留失败目录。修正为CuTe自身的线性索引，并用继承原Copy_Traits的记录器直接捕获每次Copy_Atom调用的源/目标指针作对照，而不是手写lane映射或只检查总元素数。

保持v128的数学、旧factor先缓存、32次barrier、三stage、50688B shared、64×128×128 CTA、128线程、32 partial寄存器及原guard/fallback。
不是旧v91全部地址重物化、v46全局row cursor或v131只换加载顺序；本轮只修复v128增加的A迭代器寻址。

沿用原门槛，不因失败放宽：相对v89循环最多+5%、allocated≤168、热local≤1、64条原生INT4/16LDSM/9copy/1barrier；8次下一组A加载后仍有至少8次旧partial加权。新增要求：shuffle不能进入热循环。
原控制完整编码必须不变。通过后再检查实际3CTA、GPU正确性/MSE、安全性及24样本三轮1000/200交错配对；正向才补四模式。
失败则停止这一修复，不扫相邻缓存地址数或寄存器上限。当前未宣称新性能或MSE结果。

## A100结果

源提交`9795c7d2bd68d8d6b495df1464e0d3028df5e34d`先推送GitHub，再在A100 fetch/ff-only同步编译。原v89控制完整编码保持一致。
CPU真实CuTe Copy_Atom核对通过3072个源地址、12288个目标word；这是布局验证，不是GPU数值/安全性验收。

| 整数热循环指标 | 最佳v89 | 原候选v128 | 寻址修复v134 |
|---|---:|---:|---:|
| 静态指令/G128 | 323 | 374 | 354 |
| 相对最佳工作量 | 1.000× | 1.158× | 1.096× |
| S2R | 2 | 8 | 2 |
| LOP3 | 4 | 21 | 10 |
| Allocated registers/thread | 168 | 168 | 168 |
| 静态活跃寄存器峰值 | 166 | 153 | 158 |
| 热local load/store | 0/0 | 0/0 | 4/0 |
| 原生signed/unsigned INT4 MMA | 32/32 | 32/32 | 32/32 |
| LDSM / async copy / barrier | 16/9/1 | 16/9/1 | 16/9/1 |

地址缓存确实减少了重复坐标计算：S2R回到2、LOP3减少11条，总静态374→354。但仍高于最佳9.60%，且额外长存活地址伴随4次热local reload；不能只看减少了20条指令就判断更快。
8次下一组A加载后仍有32次旧partial加权，shuffle没有进入热循环，原生两路INT4/供数审计通过。
原“最多+5%工作、热local≤1”两项投入门槛仍失败；**停止此修复，不进行候选GPU性能测试或相邻iterator扫描**。
初次CPU shape失败及修正版编译证据全部保留，没有删掉失败记录或放宽门槛。

当前GEMM最佳仍为O3 v89、O7/O8 v78；没有新增Event延迟、MSE、CV、NCU、sanitizer或实际occupancy结果，不能写成实测慢9.60%或“优化收益0%”。
正式扩展SHA仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
本轮只说明这项依赖隐藏实现尚不能以足够低的寻址/资源成本完成；并不证明所有跨K流水线方案无效。

[26份原始文本及SHA](evidence/a100_o378_roof_v134/README.md)。复核命令：

```bash
python -m pytest tests/unit/test_o3_cached_a_iterator.py tests/unit/test_roof_v134_evidence.py tests/unit/test_o78_mma_phase_stalls.py -q
```

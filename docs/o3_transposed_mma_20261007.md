# O3：交换原生 MMA 操作数的独立候选

日期：2026-10-07；轮次v124。目的仍是降低GEMM工作与依赖开销，不调整精度或正式默认。

## 查重与思路

固定尺寸已经在v45做过，取消重测。v47只是warp长宽比，v85重排物理payload，v88改变atom遍历；v107保留A×Wᵀ，仅交换哪个片段常驻。本轮不同：**以W作为左操作数，计算W×Aᵀ的fragment，再按转置坐标直接写原Y**。

保持原逻辑输出CTA64×128、4warp、三stage、8条MMA链、32个partial/64个accumulator槽位。内部MMA的M/N变为128×64，W沿两段M64流入，激活low/high作为右操作数常驻。原input/scale/metadata buffer、group-major布局、grouped CTA遍历、guard/fallback均保留，无额外转置buffer/kernel。低路用原生S4×U4，高路S4×S4，仍然是两路INT4。

预期每线程独有W列/scale从16降为8；需要由实际CuTe坐标和机器码确认。总MMA和LDSM工作预计不增加，FP32最后恢复scale的顺序不变。但是fragment/寄存器分配和最终store组织会改变，**不能描述成严格单因素scale缓存实验**。减小源码中的scale集合也不保证提速。

## 运行前固定门槛

先在CPU验证整数乘法交换、真实CuTe输入/输出坐标；只在A100项目内编译和反汇编。比较同binary的原v89控制完整编码。

- 同一正式候选entry：32条S4×S4＋32条S4×U4/G128，不得是INT8。
- 每G128仍16条LDSM、9条cp.async对应copy、一处CTA barrier。
- allocated registers≤168，无热local load/store。
- 整数热循环静态指令至少减少3%，或活跃寄存器峰值至少减少16；只减少少量scale load不够。
- 原控制SASS、转换、元数据、guard、量化语义不变。

未通过则停止，不扫描相近transpose/tile参数，不启动候选GPU，也不编造MSE或性能成绩。通过后再做实际资源、GPU数值及安全检查，直接24样本三轮配对，1000/200预热/测量；GEMM正向才补四模式端到端。不做小规模性能筛选。

## A100审计结果：停止该方向

源码commit `cad527394266c7a4ee0fe1f7a91164ac23b33a38`先在本地提交/推GitHub，再在A100项目目录fetch并ff-only同步。CUDA12.8、原pinned CUTLASS不变。真实CuTe host检查覆盖8192个输出的唯一owner及49152个输入坐标，确认W scale集合16→8；这不是GPU数值验收。

| 指标，整数G128热循环 | 原O3 v89 | 转置MMA候选 |
|---|---:|---:|
| 静态指令 | 323 | 322（仅−0.31%） |
| 活跃GPR峰值 | 166 | 160（仅−6） |
| 分配寄存器/线程 | 168 | 168 |
| S4×S4＋另一条原生INT4 | 32＋32 U4×S4 | 32＋32 S4×U4 |
| LDSM / async copy / CTA barrier | 16 / 9 / 1 | 16 / 9 / 1 |
| 普通IMAD（包括64个加权累加） | 65 | 65 |
| factor读取 | 8条LDS.64 | 8条LDS |
| 热local load/store | 0 / 0 | 0 / 0 |

两侧循环另外各有3条 `LDS RZ,[RZ]`，不是有效factor载入。旧控制完整SASS编码一致，同候选entry确认原生S4×U4、S4×S4及cg async copy，没有INT8 MMA退化。

**为什么scale集合减半却没有显著减少指令？** 原先16个值由8条64-bit load读取；现在8个值由8条32-bit load读取，发射次数没有减少。每个输出的加权累加仍要做一次IMAD，MMA/供数/同步工作保持；此外 `IMAD.SHL.U32` 23→17、`IMAD.U32` 3→2，却有 `SHF.L.U32` 44→50。源码局部数据量下降被实际指令组织抵消，不能据此预测可观提速。

热循环无spill不等于整个entry无spill：两侧资源报告均有16B stack；候选ptxas记录12B spill store/load，未用它冒充热循环访存。也未调用GPU占用率API，因此不新增“实际3 CTA驻留”测量结论。

未达到预设3%指令减少或16活跃GPR减少的投入门槛；其他结构项均通过。**停止，不运行候选、不扫描相近transpose/tile参数，也不迁移O7/O8。** 这不是实测慢0.31%，更不是确认加速0.31%。没有新增Event、MSE、conversion、端到端、NCU或GPU安全验收；已验证的最佳O3 v89、O7/O8 v78保持。正式扩展SHA与5090版本不变。

本地与A100各13项CPU/旧证据回放检查通过。16份原始文本、编译命令、生成文件、源码SHA和原始审计已冻结在[v124证据目录](evidence/a100_o378_roof_v124/README.md)，可在CPU重算；二进制留在项目内归档。

本轮insight：**应区分“少了几个scale数值”与“少了多少实际load/加权指令、改变了多少资源容量”。** 当前O3的主要MMA与整数后处理工作未减少，优化仍未达到有效吞吐上界；不能用一个布局指标改善宣布瓶颈已被消除。

```bash
python -m pytest tests/unit/test_transposed_mma_codegen.py -q
python scripts/probe_transposed_mma_codegen.py --output reports/o378_roof_v124_codegen
```

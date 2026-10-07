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

## 状态

源码与CPU契约已编写，A100编译/审计尚未执行；没有新运行时间或MSE结果。当前最佳与5090不变。

```bash
python -m pytest tests/unit/test_transposed_mma_codegen.py -q
python scripts/probe_transposed_mma_codegen.py --output reports/o378_roof_v124_codegen
```

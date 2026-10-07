# O3：高低路分别加权的依赖消除候选

日期2026-10-07，v125。**独立候选，尚未通过GPU正确性或性能验收，不改默认。**

当前v89每条链是high MMA→high MMA→×16→low MMA→low MMA→乘factor并累加。本轮保持32个partial寄存器，将其分为16个low和16个high；每个M片段有4+4条独立、长度为2的MMA链。随后分别更新同一full-K accumulator：

```text
acc += low_dot  × factor
acc += high_dot × (16 × factor)
```

×16作用在可复用的列factor上，不再插入两类MMA之间。原quantizer、G128、三stage、grouped CTA64×128×128、4warp、原始输入/metadata/转换、guard/fallback以及FP32 epilogue均保持。没有新的scale表示或转换误差。

拆开的项可能各自超出INT32，即使最终和在范围内。因此两次更新明确使用PTX `mad.lo.s32` 的低32位语义，高factor通过无符号×16生成；模2³²代数保证合并后等于原先被guard证明可表示的整数prefix。禁止依赖C++有符号溢出或加入饱和。CPU含抵消/边界/随机代数检查，GPU仍须另行验收。

该指令取乘积低位后与32位C相加；不是饱和乘加，依据[NVIDIA CUDA12.8 PTX mad定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#integer-arithmetic-instructions-mad)。

查重：v67先重构`low+16high`再乘factor；v78/v79用high作为low MMA的C；v96/v97只尝试跨N64交错；v115使用INT16 packing/DP2A。本轮没有这些重构或packing，直接用两个低32位乘加更新同一整数accumulator。不是“把high移到循环外”——每G128的两路乘积和各自scale仍完全保留。

预先固定编译投入门槛：同entry两条原生INT4、32+32 MMA，16 LDSM、9 async copy、1 CTA barrier，旧控制完整编码一致；allocated≤168、热local=0；机器码确有32条同符号两MMA链且静态未完成链峰值≥8，循环静态指令增幅≤8%。此门槛检查是否值得测试，不预测加速。

门槛通过才查实际资源、GPU数值/guard/安全性，并直接做24样本×3轮、warmup1000/repeats200配对，不做小规模性能初筛。确认GEMM收益后再补四模式。失败即停止，不扫相邻链/排布、不迁移O7/O8、不修改5090。

```bash
python -m pytest tests/unit/test_split_weighted_codegen.py -q
python scripts/probe_split_weighted_codegen.py --output reports/o378_roof_v125_codegen
```

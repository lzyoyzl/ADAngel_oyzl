# v75：当前全K kernel 的寄存器存活检查

**没有新增CUDA kernel、性能/MSE测量或默认切换。** 本轮检查已验证v67的同一份cubin，
目的是判断“资源报告168个寄存器只是回退分支造成的，删除它就能提高驻留”是否有依据。

## 结果

在A100服务器使用CUDA12.8的`nvdisasm --print-code --life-range-mode count`，
按同一entry的向后跳转识别两个G128主循环，并核对各含32条U4×S4和32条S4×S4。

|同一函数中的区域|静态循环指令数|同时存活GPR峰值|循环内local访问|
|---|---:|---:|---|
|全K整数路径，PC `0xa70–0x2200`|378|160|无LDL/STL|
|原逐组FP32回退，PC `0x4c10–0x6730`|435|165|2条静态LDL|

整个entry分配168 registers/thread。表中为**当前二进制的静态存活分析**，
不是动态寄存器使用采样，也不是所有等价实现的寄存器最小值。
主循环的静态指令数不等于每次循环必定执行的指令数；其中包含条件搬运/出口控制。

A100每SM有65,536个32位寄存器。当前128线程/CTA若要求4个CTA同驻留，
仅考虑寄存器容量，门槛已经是 `65536 / (4 × 128) = 128` 个/线程。
现有整数热路径峰值160，明显高于128。**不能据此预期只去掉回退分支就能变成4 CTA/SM**；
需改变热路径的存活安排或引入spill，且重新编译后的驻留与性能必须实测。
寄存器数只是必要条件，还要满足shared memory、线程等约束，4 CTA也不是速度保证。

## 对下一步的影响

不开展仅删回退、强压寄存器上限的枚举。也不重复以零spill为目标：
[v70 NCU](../a100_o378_roof_v70_ncu/README.md)已经确认被profile的全整数路径动态local访问为0。

若进一步缩短fragment存活，必须同时保留独立MMA链和A/B复用。
之前[v26 N64 tile](../a100_o378_roof_v26/README.md)虽达到4 CTA，却将LDSM工作增加50%，实测变慢；
[v63合并partial](../a100_o378_roof_v63/README.md)也没有确认收益。
它们不是新整数路径一定失败的证明，但足以否定“少寄存器/更高occupancy必然更快”的筛选方式。
后续新候选需要有不同的、可检查的动态工作或生命周期改善，不能只重复这些配置。

本轮没有新的吞吐提升、MSE或sanitizer结论，当前最佳不变。
0.220347ms仍是原容量模型的必要MMA下界，不能因暂未找到有效候选而把目标改成已达成。

## 证据与复现

- 工具脚本本地实现、push后，A100 fetch/ff-only到`5f83d31563c67ecc2377fa0f117b77a9d66b3f3e`。
- 原cubin SHA：`e10004af910b2148020b1d55789886ad542f86cd798ab2450e25569c9f1a5e30a`。
- 原正式扩展SHA仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
- 归档SHA：`6502e0b1177b876b47aa7937ec6e02279808cd2df3e71014045e975806fc18d1`。
- [原始nvdisasm输出](reports/o378_roof_v75_liveness/o78_fullk_liveness.txt)、
  [分析与完整命令](reports/o378_roof_v75_liveness/analysis.json)均保留。

```bash
python scripts/inspect_o78_register_liveness.py --output reports/v75_recheck
python -m pytest tests/unit/test_o78_register_liveness.py tests/unit/test_roof_v75_evidence.py -q
```

解析器以精确entry、两个完整主循环及count格式为门槛，缺少证据时失败而非猜测。
离线测试重算峰值/指令/范围，检查源脚本和原二进制身份。

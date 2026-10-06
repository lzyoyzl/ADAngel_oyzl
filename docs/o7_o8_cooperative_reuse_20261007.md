# O7/O8：扩大 CTA 合作范围的编译检查（v111）

结论：**未通过预设潜力门槛，停止；当前最佳和正式默认不变。** 本轮只完成一个固定候选的
CPU/CuTe 映射验证、A100 编译及指令审计，没有运行候选 GEMM，也没有新的性能或 MSE 结果。
不继续相邻 tile/warp 扫描，不将静态指令变化描述为实测加速或减速。

## 1. 先按机制查重

|已有机制|已知代价|本轮的实质区别|
|---|---|---|
|v57/v83：四 warp 扩大输出 tile|每线程 accumulator 增加，寄存器达 255、驻留减少|同时扩大合作线程数，保留每线程 64 个最终 accumulator|
|v98/v104：相同输出 tile 增加 warp|片段读取增加，完整配对测试退化|输出 tile 与线程数同比增加，保留每 warp 16 条 LDSM|
|v109：warp 私有供数|输入读取翻倍，循环及热 spill 增加|继续 CTA 共享输入，不改为 warp 私有读取|

实现历史中未找到本轮的 **128×192 输出 tile、12 warp、4×3 warp 布局**。
它不是把已测大 tile 换一个版本号重测；但编译结果不佳后，也不扩展该机制的相邻尺寸。

## 2. 固定候选及代价

|项目|v78 控制|v111 候选|
|---|---:|---:|
|CTA tile|64×128×128|128×192×128|
|线程 / CTA|128|384|
|每线程最终 accumulator / partial|64 / 32|64 / 32|
|MMA 独立链 / warp|8|8|
|pipeline stages|2|2|
|dynamic shared bytes|34304|59904|
|4096³ 的 CTA 数量|2048|704|

数学仍为两路原生 INT4、高位乘 16 后合并低位、每个 G128 独立 factor、全 K 精确整数累加。
新 tile 的输入复用增加：逻辑输入及 factor 读取量约为原来的 **59.24%**。
这是代码工作量模型，不是实测 DRAM 流量，也不是 40.76% 的性能收益。

N=4096 不能被 192 整除，末尾 tile 有 64 个有效列、128 个补齐列，故 MMA 工作量增加
**3.125%**。使用带 `ignore-src` 的 cp.async 清零补齐部分，源指针保持有效，输出写回有边界检查。
该清零语义见 [CUDA 12.8 PTX 文档](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)。

计划以一个 12-warp CTA 替代三个 4-warp CTA，保持 12 个驻留 warp；**本轮未进行 CUDA
occupancy 查询或 GPU 执行，不能把计划驻留数当作实测结果**。

## 3. 编译审计结果与停止依据

A100、CUDA 12.8.93、固定 CUTLASS commit。控制 entry 与原 v78 的 SASS 机器码一致。
下面比较的是各 entry 的一个 G128 整数热循环，而非 kernel 延迟。

|编译指标|v78|v111|
|---|---:|---:|
|分配寄存器 / 活跃峰值|168 / 166|168 / 162|
|静态循环指令数|383|387|
|S4×S4 / U4×S4 MMA|32 / 32|32 / 32|
|LDSM / 异步复制|16 / 10|16 / 8|
|热 local load/store|0|0|

复制指令减少，但较大合作布局的地址计算、边界判断及调度没有减少整体指令工作。
按全矩阵 warp 数量计入补齐，静态指令工作量比为：

```text
(704 × 12) / (2048 × 4) × 387 / 383 = 1.042020
```

也就是增加约 **4.20%**，不满足编译前规定的“至少减少 5% 静态工作量”门槛。
这不是证明真实 kernel 必然慢 4.20%；它说明：在当前已知的指令/计算瓶颈下，
减少逻辑搬运没有转化成足够的编译潜力，不值得继续投入完整 24 样本测试。
不放宽门槛、不为该候选继续集成 fallback，也不迁移到 O3。

## 4. 正确性与结果边界

- Host-only CuTe 验证：24576 个输出元素各有一个 owner、12288 个 vector-store pair、
  partial/最终坐标映射一致、20480 个布局字节 owner 及 40960 次 nibble 检查，全部通过。
- CPU 验证：所有有效输出所依赖的旧 64×128 guard 均被新 CTA 的聚合 guard 覆盖。
- 同一候选 entry 含原生 S4×S4 和 U4×S4，无 INT8 MMA 替代；无编译 spill。
- **候选没有完成 fallback 集成，非零 guard tile 只返回，不能用于实验或正式运行。**
  它是独立编译原型，不是可替换后端；GPU 数值、MSE、内存安全及四模式验收均未完成。

正式扩展 SHA 前后相同：
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
O3 v89、O7/O8 v78+v73 的最佳记录保持不变；量化、转换、正式默认、5090 均未修改。
本轮没有新的 GEMM 或端到端提升，优化目标尚未达到。

## 5. 证据与复算

- [源 SHA、命令、编译 gate 和完整产物索引](evidence/a100_o378_roof_v111/reports/o378_roof_v111_cooperative_reuse_codegen/codegen.json)。
- [编译日志](evidence/a100_o378_roof_v111/reports/o378_roof_v111_cooperative_reuse_codegen/build.log)、
  [资源报告](evidence/a100_o378_roof_v111/reports/o378_roof_v111_cooperative_reuse_codegen/resources.txt)、
  [SASS](evidence/a100_o378_roof_v111/reports/o378_roof_v111_cooperative_reuse_codegen/o78_cooperative_reuse.sass)。
- [冻结证据说明](evidence/a100_o378_roof_v111/README.md)。完整二进制归档 SHA256：
  `a991ae23cbeffa336b5165da5f8027da46a3409c6097610cf0d402ce92026f16`。

不启动 GPU 的复算：

```bash
python -m pytest tests/unit/test_cooperative_reuse_codegen.py \
  tests/unit/test_roof_v111_evidence.py -q
```

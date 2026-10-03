# A100 O3 / O7 / O8：暂停前最佳版本汇总

日期：2026-10-03。范围：本任务的A100优化；RTX 5090后端未修改。

**结论：保留O3 v79、O7/O8 v78 GEMM + v73在线准备。**
它们是已验证的独立优化候选，**尚未替换正式默认**。最新v88没有确认稳定收益，不纳入最佳。
本轮测试与归档完成后，按用户要求暂停，不再自动启动下一轮。

## 1. 实验口径

A100-PCIE 40GB、CUDA12.8；24个Llama-2-7B真实权重/激活样本，`M=N=K=4096`，FP32输出。
每次50预热、200次测量；转换阶段在一个CUDA Event区间内重复100次后摊销，
GEMM和端到端总时间为单次直接Event计时。所有离群和CV失败保留。

|后端|源数据与计算路径|输出MSE参考|
|---|---|---|
|O3|MXFP4-G128权重→Q4；原INT8激活→低U4/高S4；两路原生INT4|O0|
|O7|NVFP4-G128权重→Q4；MXFP8 E4M3-G128激活→Q8，再拆两路INT4|O5|
|O8|HiF4-G128权重→Q4；FP6 E2M3-G128实验变体激活→Q6并符号扩展，再拆两路INT4|O6|

O7/O8沿用已确认的实验格式定义；G128扩展及FP6两级scale不冒充标准NVFP6硬件格式。
公共FP16→源格式量化不计入转换；源格式→定点/平面/最终布局及必要scale、factor、guard准备计入。
不同后端的MSE参考不同，不能据此直接排名格式的普遍精度。

## 2. 当前最佳性能

下面均为24样本median。O3来自v79、O7/O8来自v78，**不是三者在同一轮重新联合测量**。
三轮纯GEMM与独立四模式复测分开列出，不能挑较小值拼成一轮，也不能相加阶段median构造端到端。

### Compute-only / GEMM-only

|后端|三轮GEMM median ms|相对同轮上一版的配对吞吐提升|独立四模式复测GEMM ms|
|---|---:|---:|---:|
|O3 v79|0.447488|+1.83%，对照v61|0.466432|
|O7 v78|0.459008|+0.89%，对照v67|0.471552|
|O8 v78|0.461824|+0.89%，对照v67|0.476160|

上述提升不是相对原始正式版本的累计提升；不将不同历史轮次的百分比相加或相乘。

### Conversion-only：所需在线准备的批量摊销开销

|后端|总转换 median ms|包含内容|
|---|---:|---|
|O3|0.047555|A/W转换、布局及W factor/范围检查|
|O7|0.066883|A/W转换、布局、行factor/anchor/范数和CTA范围检查|
|O8|0.091581|同上，按HiF4/FP6源格式处理|

这是当前全K候选的完整准备成本，**不声称它是全部历史版本中转换单项最快者**；
较早不带full-K metadata的转换路径成本更少，但没有当前GEMM的全部前提。

### Cold：每次在线准备A/W，再执行GEMM

|后端|Cold total median ms|
|---|---:|
|O3|0.531968|
|O7|0.551936|
|O8|0.577024|

### Steady-state：缓存权重准备，在线处理激活并执行GEMM

|后端|Steady-state total median ms|
|---|---:|
|O3|0.499712|
|O7|0.524288|
|O8|0.540672|

Cold/steady均直接测量整段，不等于conversion median与GEMM median之和。
O3最新一次steady配对收益区间跨1，因此不宣称其相对上一版steady已经稳定提高。

## 3. 输出MSE

|后端 / 参考|Median MSE|Mean MSE|
|---|---:|---:|
|O3 / O0|0.006653010287410|0.007578847013303|
|O7 / O5|0.005536172273439|0.005053635851003|
|O8 / O6|0.004411084910986|0.004381379299074|

当前O3对v61、O7/O8对v67输出逐位一致。更早引入全K整数累加时曾改变FP32舍入顺序，
通过了MSE回归；不能把“相对上一版逐位相同”说成“整个优化历史从未改变任何输出bit”。

## 4. 当前采用的优化手段

**转换与数据组织。** 直接生成G128-major packed数据和scale布局，向量化读写及packing，
避免先生成中间布局再完整重排。O7/O8的v73在行内融合转换、组平方和及factor/anchor生成，
减少独立kernel启动和重复读取；CTA安全检查仍计时，没有把新准备成本移到免费离线阶段。

**GEMM。** 用`a = low_u4 + 16 × high_s4`表示激活，保留两路原生INT4 Tensor Core。
partial保存在寄存器，A/B由`cp.async`多阶段搬运，B片段跨M复用，N64寄存器片段流式计算，
最终结果仅写回一次。A100使用自己的SM80流水线，不声称支持或使用TMA。

**scale与全K累加。** 把每组有效scale精确拆为共同base和整数factor，在安全范围内执行：

```text
O3：    I += P[group] × W_factor[column, group]
O7/O8： I += P[group] × A_factor[row, group] × W_factor[column, group]
最后将 I 转成FP32，恢复相应的行/列base scale。
```

这没有丢掉独立G128 scale，也不是跨组随意乘一个scale。GPU检查系数、乘积、整数累加前缀
和FP32输出范围；不安全CTA走原逐组FP32路径，非法输入拒绝。24真实样本中，
O3/O7均走安全整数路径；O8仅`layer_24_o_proj`的12个CTA回退。

**MMA调度。** 每个N64片段保留八条独立partial链：先计算两个K64的high点积，乘16，
再把它直接作为两次low MMA的累加输入。减少独立high/low合并工作，同时保留指令级并行。

|资源|O3 v79|O7/O8 v78|
|---|---|---|
|CTA tile / 线程数|64×128×128 / 128|64×128×128 / 128|
|pipeline stages|3|2|
|Shared/CTA|50,688 B|34,304 B|
|寄存器/线程；最大CTA/SM|168；3|168；3|
|spill|有；入口28 B load / 28 B store|0|

## 5. 本轮v88结论与剩余差距

本轮实际调用CUTLASS/CuTe的整个fragment MMA遍历，替代手写atom顺序。
24样本×3轮中O7/O8配对吞吐点估计+0.67%/+0.45%，95%区间均跨1；
虽然主循环383→377条、MMA复用标记25→27，但未证明稳定加速，**不替换上述最佳**。
输出逐位一致；同entry原生INT4审计和有限memcheck/synccheck/racecheck通过。

已有模型给出必要MMA容量下界约**0.220347 ms @1410MHz**，不是保证当前kernel可达的时间。
当前GEMM仍约0.45–0.46 ms，尚未达到接近该下界的目标。必要MMA/fragment供数未减少，
逐组整数系数乘加和流水线控制仍存在，168寄存器限制驻留数；现有NCU也显示发射等待。
不能把不同run的Event与NCU Duration相除，或把warp stall采样比例解释成可直接消除的耗时。

**验收边界：** 真实样本输出/MSE、同entry ISA和有限形状安全检查已通过；
共享未锁频GPU上的直接计时仍大量CV≥3%，不能声称严格全阶段稳定性达标。
sanitizer覆盖较小M/N和完整K4096，不外推为所有4096³输入安全证明。
正式默认不变、目标未宣告完成；本次仅按用户要求暂停。

## 6. 证据与恢复入口

- [O3 v79：性能、MSE、资源与四模式](evidence/a100_o378_roof_v79/README.md)
- [O7/O8 v78：性能、MSE、资源与四模式](evidence/a100_o378_roof_v78/README.md)
- [O7/O8 v73：在线转换与metadata融合](evidence/a100_o378_roof_v73/README.md)
- [本轮v88：未采纳结果及原始证据](evidence/a100_o378_roof_v88/README.md)
- [当前最佳的NCU分析](evidence/a100_o378_roof_v80_ncu/README.md)

候选入口：`benchmark_o3_eight_chain_probe.py`和`benchmark_o78_eight_chain_probe.py`，
对应A100 `reports/o378_roof_v79_codegen/`、`reports/o378_roof_v78_codegen/`；
O7/O8准备库为`reports/o378_roof_v73_codegen/`。原始计时位于对应`runs/*_trace24`和`runs/*_four24`。
恢复时使用新输出目录，保留已有结果；只有收到明确继续指令后再启动测试。

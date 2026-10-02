# v72：O7/O8 系数提前计算，未确认性能收益

本轮只测试一个独立候选：保持原全K整数实现，将 scale 系数乘法从 MMA partial 的依赖链中分离。
首层4样本×3轮未确认收益，不扩大24样本或四模式性能测试；当前最佳、正式默认、O3和5090均不变。

## 改动及编译证据

v67 C++ 原本写成 `partial * (A_factor * W_factor)`，但 NVVM/PTX 与最终SASS将它结合为
`(partial * A_factor) * W_factor`。v72用纯 inline PTX 固定系数乘法节点：

```text
coefficient = A_factor * W_factor    // 不依赖 MMA 结果
acc += partial * coefficient
```

没有近似运算、binary patch、新量化或额外预处理；factor乘积、每组贡献及所有整数前缀仍由原guard保护。
保留原v67全K控制和旧FP32回退；CTA64×128×128、4warp、2stage、两路原生INT4及epilogue均不变。
两路径都使用同一v69融合GPU准备，不能把control误记为旧tune59，或关闭control的metadata准备。

通过真实SASS寄存器定义/使用关系核对，候选64个输出更新都使用独立系数；不是只检查源码有inline PTX。
该分析是静态依赖证据，不是周期模型或动态NCU归因。

|编译指标|v67控制|v72候选|
|---|---:|---:|
|整数主循环静态指令|378|373|
|主循环原生INT4 MMA|64|64|
|主循环LDSM|16|16|
|独立于partial的系数乘法|0|64|
|先乘partial与单侧scale|64|0|
|主循环local load / store|0 / 0|4 / 0|
|寄存器/线程、驻留CTA/SM|168、3|168、3|
|ptxas stack bytes|8|40|
|ptxas spill stores / loads bytes|8 / 8|52 / 52|

4条新增local读取位于下一stage预取路径。整entry的spill数字还包含回退/epilogue，不能全算作每组热路径流量。
同entry PTX/SASS含U4×S4、S4×S4与cp.async/LDGSTS.BYPASS，无INT8替代。
控制entry完整opcode计数及总1944条指令与v67一致；GPU另做输出逐位对照，不仅依赖静态指纹。

## 配对测试结果

A100，4096³，`layer_00_{q,k,v,o}_proj`，每样本3轮，warmup50/repeats200。
48条compute-only记录、9,600次直接GEMM Event计时全部保留；每组的GPU转换/guard已缓存。
同进程、同输入、单stream、预分配计时资源、交错A/B顺序。不锁频，不筛除离群或CV失败。

|后端|原全K控制 ms|系数提前候选 ms|配对吞吐变化|speedup描述性95% CI|CV≥3% 控制/候选，各12条|
|---|---:|---:|---:|---|---|
|O7|0.450048|0.450816|−0.67%|[0.989807, 1.015982]|12 / 9|
|O8|0.445440|0.453120|−1.83%|[0.968820, 1.001119]|12 / 11|

延迟为样本内三轮median后再跨样本median；吞吐变化由配对比值汇总，不能直接从两个总体median反推。
两组区间均跨1，且大多数记录CV超标，结论是**未确认收益**，不是精确证明慢了固定百分比。
本轮没有新的conversion/Cold/steady性能数据，也没有新增NCU；不能确定新增local读取单独造成了多少损失。
MMA/供数工作量和驻留CTA未改善，因此不能仅因依赖链缩短就声称更接近0.220347ms容量下界。

|输出MSE参考|控制和候选共同Median|共同Mean|候选相对v67输出差MSE|
|---|---:|---:|---:|
|O7 / O5|0.000102067943157889|0.0000991674929848502|0|
|O8 / O6|0.000123493261233427|0.000122317652712578|0|

48条输出均与原v67全K逐位相同。这是首层4样本的MSE，不能与24样本总MSE比较为精度提高。

## 正确性、安全性与停止依据

- 29项CPU候选/guard检查通过；另增3项离线证据测试重算所有计时、配对汇总、MSE一致性、SASS依赖和二进制SHA。
- 64项GPU四模式数值检查：随机、全零、正负极值、宽scale触发回退，2后端×2实现×4模式；非默认stream。
- 12项边界检查：非法编码拒绝、整数范围、饱和范数、零scale和FP32 epilogue范围等。非法输入仅准备/拒绝，不执行GEMM。
- 有限memcheck、synccheck各复跑64+12项，均0 errors；M/N最大128/256、K4096，不是4096³ sanitizer或新racecheck。
- 正式扩展SHA始终为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，未重编译或切默认。

该依赖调整在机器码层面生效，却没有形成实测收益。停止这个候选，不继续枚举相邻调度位置或寄存器限制。
后续需要能减少实际工作量/改善真实等待的新依据，而不是将同一乘法重新排布当成确定加速。

## 证据与复现

本地实现/GitHub推送/A100 fetch+ff-only：编译提交`7b0aa28`，测试提交`da4926b`。

- [原始Event与MSE](runs/o378_roof_v72_screen/results.jsonl)、[汇总](runs/o378_roof_v72_screen/summary.json)。
- [完整SASS、PTX及receipt目录](reports/o378_roof_v72_codegen/)、[整数主循环依赖复算](reports/o378_roof_v72_codegen/loop_analysis.json)。
- 编译归档SHA：`0c56bf4670896607fe2efd18b74c818dd97eb749217b79714dd57f876683bca6`。
- 运行/安全检查归档SHA：`46513c3b67cf6e61919d8aad97865e03edb9581c4f1cae7223a3ea918fe6840a`。

```bash
python scripts/probe_o78_coefficient_codegen.py --output reports/v72_rebuild
python scripts/benchmark_o78_coefficient_probe.py \
  --cubins reports/v72_rebuild --output runs/v72_recheck \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
python -m pytest tests/unit/test_o78_coefficient_candidate.py tests/unit/test_roof_v72_evidence.py -q
```

输出目录须新建。既有v67 cubin/driver、v69准备库均校验receipt与源码SHA，不能跳过身份核对。

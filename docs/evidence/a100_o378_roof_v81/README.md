# v81：完整 warp 搬运 scale 因子，未确认 GEMM 收益

**结论：停止此候选，不扩展到24样本，不替换当前最佳或正式默认。**
v80 NCU 定位到的 metadata excessive wavefronts 已消除，但四样本三轮配对
O7/O8 GEMM吞吐变化仅 +0.57% / −0.57%，置信区间均未确认正收益。
这是一次有明确热点依据、但没有产生有效加速的优化，不算新最佳。

## 唯一改动

v78 用半个warp（16线程×16B）搬运一块256B的A factor；v81改为warp0全体
32线程×8B，W factor由warp1的32线程×16B搬运。每CTA/G128仍为256B A +512B W，
没有新增padding、重复数据或预处理。8B `cp.async` 使用 `.ca`，原16B使用 `.cg`，
所以缓存行为也有变化，不能称为严格的单因素线程参与率实验。

其余保持v78：64×128×128 CTA、4warp、两阶段34304B shared、八条独立MMA链、
逐G128原scale、全K精确整数factor、范围保护及FP32回退、最终FP32输出。
两个对照均使用v73行级融合准备；没有修改量化、转换或正式扩展。
O3和RTX5090没有改动。

## 指令与资源

同一个正式候选entry `adangel_roof_o78_warp_metadata_candidate` 验证
U4×S4和S4×S4原生INT4，以及异步搬运；不匹配其他probe冒充审计。

|指标|v78控制|v81候选|
|---|---:|---:|
|寄存器/线程；最大CTA/SM|168；3|168；3|
|整数循环最大存活GPR|166|164|
|整数循环静态指令|383|388|
|每G128 IMMA / LDSM|64 / 16|64 / 16|
|整数循环LDGSTS|10×BYPASS.128|9×BYPASS.128 + 1×64|
|整个entry stack / spill stores / spill loads B|0 / 0 / 0|0 / 0 / 0|

v78控制的完整编码SASS与原二进制一致。降低2个活跃寄存器没有改变分配或驻留数。

## 实测：4个真实样本×3轮

A100、4096³、`layer_00_{q,k,v,o}_proj`，warmup50、repeats200、单stream、预分配，
控制/候选交错顺序。48条记录全部保留，没有过滤CV失败或挑选最快轮。
ms为样本内跨轮median后跨样本median；吞吐变化先按样本配对，不等于两列ms直接相除。

|后端|v78 GEMM ms|v81 GEMM ms|配对吞吐变化|speedup描述性95% CI|CV≥3%，控制/候选（各12条）|
|---|---:|---:|---:|---|---|
|O7|0.446976|0.444928|+0.57%|[1.000000, 1.009238]|12 / 11|
|O8|0.439552|0.443904|−0.57%|[0.988399, 1.002347]|12 / 11|

两组区间都未确认改善，且未通过严格CV门槛。共享、未锁频环境下的波动原样保留，
没有证据将具体离群值归因于其他进程、温度或调度事件。
**本轮只测cached compute-only；没有新的conversion、Cold或steady计时结果。**
转换实现相同不等于已经测量了端到端收益。

|四样本输出指标|O7相对O5|O8相对O6|
|---|---:|---:|
|MSE median|0.000102067943157889|0.000123493261233427|
|MSE mean|0.000099167492984850|0.000122317652712578|
|候选相对控制输出MSE / 最大绝对差|0 / 0|0 / 0|

48条输出均对v67数值参考及v78控制逐位一致，finite FP32，payload/scale metadata精确一致。
这是首层四样本，不是24样本精度改善。另有64项小M/N、完整K4096验证及12项边界检查，
覆盖原FP32回退；memcheck、synccheck均0 errors，racecheck为0 errors/0 warnings。
Sanitizer覆盖64×128与128×256输出，不冒充4096³ sanitizer验收。

## NCU：热点改善没有变成延迟改善

只对O7的`layer_00_q_proj`采集一次完整NCU；50次目标预热后捕获一次，
`--set full --cache-control all --clock-control none`。与v80采用相同协议，
symbol、cubin、静态指纹、动态工作和输出均核对；2048个CTA全部走安全整数路径。
下表不是交错Event测速，不能据NCU Duration计算正式加速比；没有新O8 NCU结果。

|指标|v80 O7（v78）|v81 O7|
|---|---:|---:|
|NCU Duration ms|0.392800|0.393408|
|Source shared wavefronts|38,141,952|30,801,920|
|Source shared excessive wavefronts|6,422,528|0|
|Source global excessive sectors|3,663,610|0|
|原始L1/TEX wavefront容量服务下界 ms @1410MHz|0.182582|0.188704|
|原始shared wavefront子集服务下界 ms @1410MHz|0.164753|0.170767|
|动态warp指令|105,521,152|106,823,680|
|动态IMMA / LDSM|16,777,216 / 4,194,304|相同|
|Eligible warps / scheduler|0.747220|0.767502|
|Issue active %|47.0028|47.5790|
|Achieved occupancy %|17.9421|17.9679|
|动态LDL / STL|0 / 0|0 / 0|

Source派生wavefront/sector不是DRAM字节，也不同于容量模型使用的原始硬件计数器。
这次消除了目标source派生excessive计数，但原始硬件wavefront计数推导的L1/TEX服务需求
反而增加约3.35%，不能声称总shared/L1硬件工作量下降。表中的服务下界不是可相加的实测耗时，
也不能据此精确分配性能变化的原因。动态指令增加约1.23%，必要MMA、168寄存器、
3CTA驻留限制未变。因此不能用一个计数器变好证明关键路径变短；继续微调同类metadata
copy没有足够收益依据。约0.220347ms的理想MMA容量下界不变，未接近目标。

## 复现与追溯

候选源码提交`34c1651`，定向NCU入口提交`604061d`；均先本地实现/push，再A100 fetch/ff-only merge。
正式扩展SHA保持`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
本目录保留编译/SASS/PTX、原始Event、数值与sanitizer记录、NCU文本及receipt。
完整二进制归档`tmp/o378_v81_complete.tar.gz`同时保留在本地和A100，SHA256：
`3095cc7a8c7c6c281cb34b6e07ecd655cde6beebec0695e8028d829e29ca93f6`。
`.cubin/.ncu-rep`不加入Git；源数据、计时与采集原始记录不回写。

```bash
python scripts/probe_o78_warp_metadata_codegen.py --output reports/v81_rebuild
python scripts/benchmark_o78_warp_metadata.py \
  --cubins reports/v81_rebuild --output runs/v81_recheck --samples 4 --rounds 3
python -m pytest tests/unit/test_o78_warp_metadata.py tests/unit/test_roof_v81_evidence.py -q
```

运行前需已有v67/v73/v78依赖artifact，输出目录须不存在；代码保留用于复核，不作为最佳配置。

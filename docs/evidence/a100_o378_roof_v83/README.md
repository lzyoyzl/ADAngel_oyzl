# v83：N256 复用 A 片段，未获得性能收益

**不采纳，不扩大24样本或四模式。当前最佳仍为O3 v79、O7/O8 v78+v73，正式默认与5090不变。**

## 唯一改动与计算语义

从v78的 `64×128×128` 扩为 `64×256×128`，仍4 warp、2-stage、N64寄存器片段流、
8条独立high→high→×16→low→low链，最终每个输出只写一次。
通过每份A片段服务更多N列减少重复供数；不同于v57扩大M，本轮不增加A片段寄存器。
代价是每线程最终accumulator从64增至128，launch bound由3 CTA改为2 CTA以容纳寄存器。

量化、两路原生U4×S4/S4×S4、逐G128整数factor和全K安全界保持不变，转换/guard仍使用v73。
原guard以N128划分，新CTA同时读取相邻两项；任一项要求回退，则整个N256使用原逐G128 FP32算法。
非法状态仍在host拒绝。扩大回退范围可能改变另一半的浮点舍入，本轮不声称所有输入都逐位一致。

## 四个真实样本、三轮配对

A100、原FP16 trace直接量化源格式，首层q/k/v/o、4096³；50预热、200次单次CUDA Event，
单stream、预分配、循环换序。控制/候选均通过同一个原生计时驱动。
GPU转换和metadata预先准备，此次只测cached compute-only；没有conversion、Cold或steady结果。

|后端|同轮v78 N128 ms|N256 ms|配对吞吐变化|描述性speedup 95% CI|CV≥3%，旧/新（各12条）|
|---|---:|---:|---:|---|---:|
|O7|0.449536|0.451584|−0.90%|[0.986143,1.000000]|12 / 9|
|O8|0.444416|0.450048|−1.36%|[0.983871,0.989910]|12 / 10|

延迟按样本跨轮median后再跨四样本median；吞吐先按同样本/同轮配对，因此不等于两列汇总直接相除。
48条记录及9600次Event全部保留。共享、未锁频GPU，未通过严格CV门槛；不归因某个离群值的具体外部原因。
O7没有确认改善，O8初筛方向为负。不是24样本最终结果，也不从不同run拼接最快值。

|输出MSE参考，仅首层四样本|旧/新 median|旧/新 mean|新旧输出差MSE|
|---|---:|---:|---:|
|O7 / O5|0.000102067943158|0.000099167492985|0|
|O8 / O6|0.000123493261233|0.000122317652713|0|

全部48条真实输出finite FP32、与v67/v78逐位一致，metadata逐元素检查和MSE回归通过。
这四个样本全部走整数路径；不能外推为24样本均无回退，也不能与24样本MSE汇总比较来声称精度提升。

## 资源、指令和解释

|项目|v78 N128|v83 N256|
|---|---:|---:|
|寄存器/线程|168|255|
|Driver查询最大驻留CTA/SM|3|2|
|Dynamic shared B/CTA|34304|51712|
|Stack / spill loads / spill stores B|0 / 0 / 0|0 / 0 / 0|
|整数G128循环静态指令数|383|671|
|每CTA/G128原生MMA|64|128|
|每CTA/G128 LDSM|16|24|
|每CTA/G128 LDGSTS|10|14|
|4096³ CTA数量|2048|1024|

按相同输出工作量折算，LDSM条数下降25%，LDGSTS条数下降30%，静态主循环总指令数下降12.40%。
这些是SASS及launch几何的工作量模型，不是新NCU动态流量/耗时分解。
必要MMA和逐输出的系数乘加没有减少；寄存器增加、驻留warp由12降至8，可能抵消供数复用收益。
这是代码/资源支持的解释，**本轮没有新增N256 NCU，不能给出各机制的因果耗时比例**。

同一候选entry的PTX/SASS均通过两路原生INT4与cp.async审计，无INT8替代或热路径local访问；
包含的v78控制入口编码逐条不变。首次审计在nvdisasm存活区间导出失败：
`Invalid register count : '255'`。保留失败日志，使用已经成功导出的PTX/SASS/resource继续审计；
没有修改机器码或重编译来隐藏问题，**候选的逐指令寄存器存活报告仍不可用**。

## 验证范围与材料

16项GPU检查：O7/O8、随机/全零/正负极值/宽scale、两个配置，形状128×512×4096，非默认stream。
对FP64语义参考与v67均满足rtol/atol=1e-3；无回退时要求逐位一致，宽scale覆盖混合回退。
memcheck、synccheck均为0 errors；racecheck为0 errors/0 warnings，各自执行上述16项检查。
不得将该有限形状范围描述为4096³ sanitizer验收。
宽scale的O8候选实际出现非逐位一致，但通过语义容差：N256扩大了FP32回退范围，
与前述设计预期一致；这不影响本轮四个真实样本的逐位/MSE一致结论。
validation中的CTA计数来自原N128 guard网格，不能当作N256实际launch数量。

实现/编译commit为5187d7f；da41b2b恢复审计而不修改device源码；12a6993增加独立验证/配对计时。
先本地实现、push GitHub，再由A100 bundle fetch/ff-only merge同步。
正式扩展未重新编译，性能负向后不继续tile、stage或warp枚举。

```bash
python scripts/probe_o78_n256_codegen.py --output reports/n256_rebuild
python scripts/benchmark_o78_n256_probe.py --cubins reports/n256_rebuild \
  --output runs/n256_recheck --samples 4 --rounds 3 --warmup 50 --repeats 200
```

复现依赖原v67/v73/v78构建与trace；输出目录须新建。此脚本明确不支持端到端性能测量。

原始材料为 `runs/o378_roof_v83_screen/` 和 `reports/o378_roof_v83_*`。
本地归档保留Event、provenance、编译/安全日志及PTX/SASS，不纳入cubin/so二进制。
原完整压缩包 `tmp/o378_v83_complete.tar.gz`，服务器/本地SHA-256一致：
`b6645ca5da3c712489eb54a00dc65d575467e92f87d4c78aea2f3e158780e504`。
正式 `_sm80.so` SHA-256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
离线复核：`python -m pytest tests/unit/test_o78_n256_codegen.py tests/unit/test_roof_v83_evidence.py -q`。

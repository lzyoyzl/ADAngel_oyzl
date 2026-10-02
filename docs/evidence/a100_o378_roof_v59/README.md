# v59：O3 全 K 整数累加的 factor 预计算与异步供数

**24样本配对吞吐提高2.49%，已测输出/MSE逐位不变；仅作为独立GEMM候选保留，暂不改默认。**
本轮只实现一个候选，先编译审计、合成验证及四样本初筛，再做一次24样本确认。
没有扩展tile、cache或指令写法扫描，也没有重启magic-bias。

## 1. 改了什么

沿用获准的O3全K整数流式累加：每列取锚点 `h=min(e[g])`，逐G128执行
`I += P[g] * 2^(e[g]-h)`，最后转换为FP32并恢复锚点与A行scale。
两路原生INT4、独立G128量化/scale、CTA `64×128×128`、4warp、3-stage均不变。
保守INT32界检查仍必须通过；不满足范围或K条件时回退原FP32路径。

v55b每CTA、每组读取两个UE8M0字节、计算指数差及factor，再写shared memory。
本候选把factor预生成在静态W元数据中，用一个warp的16B `cp.async.cg` 拷贝
512B factor面板，与A/B共用已有commit/wait/barrier，不新增同步点。
最终anchor仍保存原UE8M0 code，只是容器变成int32；不近似scale。

这测试的是“全K整数累加＋预计算/异步factor”的组合相对旧完整最佳O3/54。
**没有在本轮同时计时v55b，因此不能把全部2.49%归因于factor改动，或跨run相减求增量。**

## 2. 配对性能与MSE

固定4096³，单stream，同一native Driver/Event循环；50预热、200次测量、3轮循环换序。
表中ms为每样本跨轮median、再取样本median；speedup来自逐样本配对，不是简单相除这两列。

|范围|O3/54控制 ms|v59候选 ms|配对吞吐提升|描述性speedup 95% CI|CV≥3%：控制/候选|
|---|---:|---:|---:|---|---:|
|第一层q/k/v/o，4样本初筛|0.450560|0.438784|+2.45%|[1.018391, 1.033175]|7/12，5/12|
|完整24样本确认|0.474112|0.462336|**+2.49%**|[1.022124, 1.028571]|35/72，36/72|

确认run共144条记录、28,800个原始Event，未锁频，全部离群/CV失败保留。
方向与初筛一致，但不是严格全阶段CV<3%验收；同trace样本相关，CI只作描述性比较。

|24样本输出MSE / O0|控制与候选相同|
|---|---:|
|Median|0.006653010287409885|
|Mean|0.007578847013302749|
|候选与O3/54输出之间的MSE|0|

72条候选输出均finite FP32、与54逐位一致。这里仅陈述已测数据，
不宣称所有未来输入在更改FP32舍入顺序后都必然逐位一致。

## 3. 审计、资源与收益边界

|编译/资源指标|O3/54控制|v55b全K|v59全K＋异步factor|
|---|---:|---:|---:|
|寄存器/线程|168|168|168|
|驻留CTA上限|3|3|3|
|动态shared分配，bytes|50688|50688|50688|
|ptxas spill store/load bytes|12/12|36/36|36/36|
|完整function静态SASS条数|944|1104|1088|

相对v55b静态指令减少1.45%，但spill未改善。静态条数不是动态工作量，也不是耗时比例。
本轮没有新NCU；不能声称某类stall下降了多少。此前v58已证明全K的MMA工作不变，
新增整数对齐和local访问会抵消I2F减少；这仍是解释小幅收益的约束，不能把省I2F视为等比例加速。
本轮尚未接近约0.220347ms@1410MHz的乐观容量下界，不保证该下界可达到。

同一正式探针entry的PTX/SASS确认U4×S4、S4×S4与异步搬运，无INT8替代。
控制的完整编码SASS与先前54/59控制相同；O7/O8 sentinel也逐条相同。
正式扩展没有重编译，SHA仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
RTX5090源码、生产调度及默认均未修改。

## 4. 不隐藏准备成本；本轮不晋级端到端默认

元数据从v55b的 `uint8[33,4096]` 改为 `int32[33,4096]`：132KiB→528KiB，增加396KiB。
相对54的128KiB原scale面板则增加400KiB。
当前独立Driver用CPU检查范围、生成factor并上传，一次权重准备wall time为
**49.93–139.58ms（24样本）**，四样本初筛为48.05–49.79ms。
它包含同步/host工作，不能当作GPU转换kernel延迟，也没有纳入compute-only。
已缓存后的重复GEMM不重复准备；缓存绑定tensor版本，修改scale会重新检查。

**因此2.49%仅是准备完成后的GEMM收益。** 本轮没有conversion-only、Cold或steady-state
的新测量，不能声称端到端获益。若将来考虑正式采用，必须把guard/factor生成改成低开销
准备路径，并将全部成本计入相应模式。本轮不为这一有限GEMM收益继续做集成，
保留完整最佳O3 GEMM54＋转换2、O7/O8 GEMM59＋转换5，不改变默认。

## 5. 验证与复现

36项合成检查覆盖随机/全零/INT8与Q4极值/零行scale/宽但安全及不安全指数范围，
K4096快路径、K256回退、非默认stream；对FP64语义参考检查，额外验证int32元数据逐项正确。
另两项scale原地修改强制回退；非法255拒绝。memcheck、synccheck、racecheck分别重跑
36项，均0 errors，racecheck另为0 warnings。
sanitizer最大形状128×256×4096；完整4096³是数值/性能测试，不混淆覆盖范围。
本地4项归档回归复算全部原始Event、汇总、MSE、审计与安全证据；不是新GPU测量。

实现：`d7793fb`；运行支持：`e1b9b07`。本地修改、push后A100 fetch/ff-only merge。

```bash
PYTHONPATH=python python scripts/probe_roof_factor_async_codegen.py --output reports/v59_rebuild
PYTHONPATH=python python scripts/validate_roof_fullk_integer_probe.py \
  --factor-async --cubins reports/v59_rebuild --output runs/v59_validation
PYTHONPATH=python python scripts/benchmark_roof_fullk_integer_probe.py \
  --factor-async --cubins reports/v59_rebuild --output runs/v59_trace24 \
  --samples 24 --rounds 3 --warmup 50 --repeats 200
```

输出目录须不存在，必须传`--factor-async`使用相匹配的元数据，不混用旧uint8元数据。
[24样本原始结果/汇总](runs/o378_roof_v59_trace24/summary.json)、
[编译审计](reports/o378_roof_v59/codegen.json)及各安全日志均保留。
归档SHA256：`abf1866e5b99fa4e16a68c57bcfb4280a475d2890f672fd7c015bb03032f1aed`。
cubin/.so留在A100，不上传Git。

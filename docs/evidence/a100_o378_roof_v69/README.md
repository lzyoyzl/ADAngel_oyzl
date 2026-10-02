# v69：融合转换与组平方和，核对全 K 候选的端到端收益

独立候选，正式默认与 RTX5090 后端不变。24样本四模式复测通过数值检查：
O7/O8 配对 GEMM 吞吐 **+6.74%/+7.01%**，Cold **+0.90%/+1.34%**，
steady **+2.19%/+2.88%**。融合保住了有限端到端收益，但不是接近理论上界的突破；
多数直接计时CV仍超过3%，不宣称严格稳定性验收通过。

## 改动及计时归属

GEMM 二进制沿用 v67：两路原生 U4×S4 / S4×S4，CTA64×128×128，
四个 warp、两级 cp.async、全 K INT32 累加与 GPU guard 回退。
独立 G128 scale、源量化、定点映射和 FP32 输出语义不变。

v68 在完成转换后重新读取约24 MiB packed A/W 来生成整数安全界。
本轮在原向量16转换的量化整数 `q` 已位于寄存器时，直接计算 `sum(q*q)`：
每8个线程处理一个G128，shuffle 归约后写一个 uint32。4096³需要总计1 MiB平方和scratch。
然后每 warp 处理一行的32组 scale/平方和，生成整数 factor、base、加权范数及行状态；
最后沿用原CTA范围判定。不再为metadata生成重新读取packed payload。

饱和UINT64范数、系数INT32界、FP32 epilogue范围与非法编码检查均保留。
安全CTA走整数路径；未满足充分范围界走旧FP32路径，不能无条件全K合并。
这是本项目的融合实现，不声称移植了某一个开源项目的完整kernel。

|模式|本轮计时内容|
|---|---|
|Conversion-only|W转换/平方和/metadata；A转换/平方和/metadata/CTA guard，分别100次摊销|
|Compute-only|以上全部缓存，只计同一个v67 GEMM|
|Cold|单次W准备+A准备+GEMM，直接CUDA Event total|
|Steady-state|缓存W，单次A准备+GEMM，直接CUDA Event total|

转换与范围检查均计入在线路径，没有用CPU oracle代替GPU工作。
所有device buffer/Event提前分配；直接端到端测量先执行，之后再单独摊销转换。
CPU oracle、源FP16到实验格式的公共量化、文件I/O不计入上述模式，与前轮口径一致。

## 四样本初筛

A100，4096³，首层q/k/v/o，一轮，warmup50/repeats200/inner100，64条记录。
控制为tune59+conversion5，候选为v67整数GEMM+本轮GPU融合准备。
下表变化是逐样本配对吞吐，而非两个汇总median直接相除。

|后端|模式|控制 ms|候选 ms|配对吞吐变化|
|---|---|---:|---:|---:|
|O7|Conversion|0.046735|0.075026|−37.75%|
|O7|Compute|0.488704|0.454144|+7.14%|
|O7|Cold|0.554496|0.549376|+0.61%|
|O7|Steady|0.527360|0.515072|+2.39%|
|O8|Conversion|0.070881|0.095724|−25.89%|
|O8|Compute|0.486400|0.456192|+6.21%|
|O8|Cold|0.568832|0.561664|+1.63%|
|O8|Steady|0.535808|0.519680|+3.01%|

不能把不同run的v68/v69延迟差声称为同轮配对加速；v68负结果仍完整保留。
本轮选择继续24样本是因为四样本端到端配对方向一致，不是只挑选GEMM收益。

## 24样本完整结果

6层×4 projection，4096³，一轮，warmup50/repeats200/inner100。
384条记录、每条全部200次原始计时均保留；同进程、单stream、交错控制/候选。
初筛与完整run独立保存，不把初筛较短的延迟混入下表。

|后端|模式|同轮控制 ms|融合候选 ms|配对吞吐变化|speedup 95% CI|
|---|---|---:|---:|---:|---|
|O7|Conversion|0.047593|0.076344|−37.62%|[0.622756,0.625688]|
|O7|Compute|0.509952|0.479488|+6.74%|[1.058065,1.072187]|
|O7|Cold|0.571648|0.567296|+0.90%|[1.007299,1.010889]|
|O7|Steady|0.547840|0.537600|+2.19%|[1.019048,1.031823]|
|O8|Conversion|0.076099|0.101404|−24.94%|[0.748804,0.751923]|
|O8|Compute|0.516096|0.482304|+7.01%|[1.067941,1.071269]|
|O8|Cold|0.599552|0.595968|+1.34%|[1.011982,1.015437]|
|O8|Steady|0.565248|0.551680|+2.88%|[1.022222,1.033708]|

转换仍不是最佳：控制不需要全K安全证明，候选需要生成factor/norm/guard。
O7 W/A转换为0.015396/0.032179→0.026040/0.050299ms；
O8为0.030495/0.045609→0.039137/0.062249ms。
新增总成本约28.75/25.31µs，消耗了约30–34µs的缓存GEMM节省，解释了Cold净收益为何很小。
这不是把各stage的median相加当作Cold；表中的Cold/steady始终为直接Event测量。

|输出MSE参考|实现|Median|Mean|
|---|---|---:|---:|
|O7/O5|控制|0.005536172426666|0.005053635833762|
|O7/O5|候选|0.005536172273439|0.005053635851003|
|O8/O6|控制|0.004411084948645|0.004381379215302|
|O8/O6|候选|0.004411084910986|0.004381379299074|

全部384条finite FP32、MSE回归与metadata检查通过。控制输出与旧tune59逐位一致；
候选允许合理舍入变化，相对旧输出max abs为1.525879e−5/7.629395e−6，
最大差值MSE为2.171332e−14/2.134541e−14。上述微小差别不应解释成量化精度提升。
O7所有CTA走整数路径；O8仅`layer_24_o_proj`的12个CTA按保守界回退，符合此前v67证据。

选定阶段CV≥3%数量，按conversion/compute/cold/steady排列，每项分母24：
O7控制`0/23/21/22`、候选`0/24/16/24`；O8控制`0/24/14/23`、候选`0/24/10/24`。
若检查任一stage，O7两条cold路径均24/24、O8分别24/24和22/24超标。
compute样本内最后50次/最初50次median之比的跨样本median，
O7控制/候选为1.107/1.126，O8为1.077/1.134，说明存在明显段内漂移，不能只归咎于少数离群。
没有锁频或额外受控实验，不能确定归因于温度、DVFS或其他负载；bootstrap区间仅作本轮描述。

## 验证与资源

- 56项CPU整数界、metadata、计时与wrapper契约检查通过。
- 64项GPU计算/四模式检查及12项边界检查通过，新增组平方和与参考逐项完全一致。
- packed A/W、旧有效scale逐位匹配conversion5；factor/base/norm/CTA状态匹配独立精确oracle。
- memcheck、synccheck在上述小M/N、K4096验证范围内均0 errors；未新增racecheck。
- 新转换Nv4/Mx8/HiF4/Nv6寄存器数为30/32/29/22；metadata各26，guard30。
  新准备路径零local/stack/spill。GEMM仍168 registers/thread、8 B spill、3 CTA/SM。
- GEMM cubin SHA由harness核对；沿用v67同entry原生U4/S4与S4/S4、异步copy审计，不是INT8替代。
- 下载后增加3项证据复算：重算448条记录的每阶段统计、完整配对汇总、MSE接受条件、
  receipt/二进制校验和sanitizer范围。连同前述CPU契约测试共59项通过。

本轮无新增NCU，不把端到端差值定量归因于某一种stall。
共享、未锁频GPU，所有离群和CV失败保留；不同轮次的最佳值不拼接。

## 证据与复测

CUDA实现commit `6f860da`，Python验证字段冲突修复与测量commit `d487388`。
正式扩展未重新编译，SHA256仍为：
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

初筛/编译/安全检查归档SHA256：
`9d697e91c5bf11b7e38331a67b4b1585eeff2bb9c54532db27dbb8703ad43a5e`。
24样本归档SHA256：`2ea75eeea082583e70e8e44222ff1e0fee4643b87b384ce6ff6a2be79759dcfc`。
[24样本原始数据](runs/o378_roof_v69_trace24/results.jsonl)、
[完整汇总](runs/o378_roof_v69_trace24/summary.json)、
[初筛原始数据](runs/o378_roof_v69_screen/results.jsonl)、
[编译receipt与源码SHA](reports/o378_roof_v69_codegen/build.json)、
[GPU验证](reports/o378_roof_v69_validation_fixed/validation.json)。

```bash
python scripts/probe_o78_fused_prepare_codegen.py --output reports/v69_rebuild
python scripts/benchmark_o78_fused_prepare.py \
  --gpu-build reports/v69_rebuild --gemm-cubins reports/o378_roof_v67_codegen \
  --output runs/v69_recheck --samples 24 --rounds 1 \
  --warmup 50 --repeats 200 --inner 100
```

输出目录必须不存在。v67 GEMM可用 `scripts/probe_o78_fullk_codegen.py` 重建，
不得跳过源码/二进制receipt核对。独立候选不会改动普通正式调度。

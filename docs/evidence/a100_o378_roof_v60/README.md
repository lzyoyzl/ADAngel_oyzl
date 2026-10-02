# v60：O3 精确GPU guard/factor准备——同步成本抵消小幅GEMM收益

**GPU准备约3.81μs；缓存后保留约2.51%的GEMM收益，但每次重新检查、回传及决策后的组合没有确认更快。**
只做一个准备路径及4样本初筛，没有扩展24样本或修改正式默认。O7/O8不变。

## 改动及安全原理

复用v59的控制/候选GEMM cubin，机器码完全不改。
新增一个GPU kernel：每线程读一列32个UE8M0 code，取anchor，生成32个int32对齐factor，
同时给出整列范围检查；每128列归约一个status，4096列仅回传128B判断信息。
不再把全部scale拷回CPU后用Python构造factor列表。

O3每G128的整数partial绝对值不超过 `128*128*8=131072`，所以
`sum(2^(code-anchor)) <= floor(INT32_MAX/131072)=16383`
足以保证每个乘积和所有前缀累加安全。
GPU在计算检查用factor时把指数差截在14：任何差≥14本来就不安全，
截断不会把不安全输入误判为安全，同时避免超范围移位。安全输入的factor完全不截断。
unsafe回退旧54；255及当前独立探针不支持的0 code明确拒绝，不执行fast GEMM。
metadata仍为int32[33,N]，量化、原G128 scale、两路原生INT4及FP32输出不变。

## 4个真实样本 × 3轮

4096³，50预热、200次CUDA Event测量、循环换序、未锁频、不删除离群。
“GPU准备”每个Event重复100次后摊销；其余每个Event单次执行。

|本轮独立计时口径|median ms|相对控制配对吞吐变化|描述性speedup 95% CI|CV≥3%，各12条|
|---|---:|---:|---|---:|
|控制54：已准备输入，仅GEMM|0.454144|—|—|11|
|缓存guard/factor后，仅v59 GEMM|0.439808|**+2.51%**|[1.016393,1.033654]|3|
|每次GPU准备＋DtoH＋stream同步＋CPU决策＋GEMM|0.452096|**−0.33%，未确认收益**|[0.990868,1.002288]|4|
|仅GPU guard/factor kernel，不含决策回传|**0.0038144**|不与GEMM比较|—|4|

ms按样本跨轮median、再取样本median；speedup先逐样本逐轮配对，所以不能用两列总median
直接相除解释符号。原始48条记录、9600个Event值全部保留，不是严格CV<3%验收。

**这些不是正式实验的四种计时模式。** 输入A/W payload已做完既有转换和布局处理。
第三行直接测量新增guard路径＋GEMM，包含CPU决策造成的GPU时间间隙，但不含既有A/W转换，
所以既不是完整Cold也不是完整steady。第四行不含回传/决策，不能据其3.81μs直接推导整体收益。
不能把它与v59的CPU准备wall time作同口径加速比；CPU准备包含更多host/同步工作。
没有新的完整conversion、Cold或steady结果。

|输出误差，仅本轮4样本，与控制相同|数值|
|---|---:|
|MSE/O0 median|0.000278282719669688|
|MSE/O0 mean|0.0002830553406440159|
|新旧输出MSE|0|

36条产生输出的记录均与54逐位一致；另12条仅测准备，metadata逐项正确。
不能把四样本MSE直接与24样本的median比较，也不把本轮称为新增24样本确认。

## 洞察与决定

50–140ms的Python准备不是GEMM的固有代价，已证明可由小型GPU kernel生成相同元数据。
但**准备计算便宜，不代表运行时检查/同步/决策便宜**。本轮加入完整新增决策链后，
缓存后约2.5%的收益消失在噪声区间中；不能只减掉准备kernel的median来虚构收益。
这是组合路径实测，不是NCU定量stall归因，本轮没有追加NCU。

范围检查只依赖静态W scale；真实缓存权重场景可在W变化时执行一次，之后复用，
不能在普通steady GEMM中反复收费。第三行有意每次重新检查，反映未缓存/Cold侧新增工作。
下一次若考虑集成，应优先融合到原W转换、避免逐次host决策；同时记录首次准备和缓存失效成本。
在GEMM收益仅约2.5%的现状下，本轮不继续为这一路径扩展实现或全量性能扫描。
完整最佳仍为O3的54＋转换2、O7/O8的59＋转换5，正式默认及5090后端未改动。

## 验证与证据

准备kernel为48寄存器、16B shared、零spill；两个GEMM cubin hash与v59完全一致。
48项GPU合成检查覆盖随机、零、极值、零行scale、宽安全指数和unsafe回退，
M/N为64×128及128×256，K4096，非默认stream；12项非法code拒绝检查。
输出与FP64语义参考比较，GPU元数据和status与CPU精确整数oracle比较。
memcheck、synccheck、racecheck各自重跑上述检查，0 errors；racecheck另0 warnings。
sanitizer不是完整4096³覆盖，后者来自四个真实样本的数值和性能检查。

本地CPU证明测试覆盖2020组指数序列及边界；归档测试重算计时/汇总并核对metadata、安全与cubin身份。
首次尝试只有编译成功，Driver因Torch primary context未激活而停止，尚未执行kernel。
修复后结果在`screen_fixed`，初次失败日志保留，不把失败运行算为成功实验。

实现commit `0b3f729`，context修复 `cfc6d7c`；均先本地修改/push，再A100 fetch/ff-only merge。
正式扩展SHA不变：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
PYTHONPATH=python python scripts/benchmark_roof_gpu_factor_probe.py \
  --gemm-cubins reports/o378_roof_v59 \
  --output runs/v60_recheck --samples 4 --rounds 3 --warmup 50 --repeats 200 --inner 100
```

输出目录必须新建。审计过的v59 cubin必须存在，不重新编译GEMM；仅编译准备kernel和Driver。
[结果汇总](runs/o378_roof_v60_screen_fixed/summary.json)、原始results与安全日志均保留。
归档SHA256：`57181029dc512551c7f79c1850640e0966f3d10b8d5188266a5d4e6d00b863b8`。

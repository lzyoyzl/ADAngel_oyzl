# v30：O3 行 scale 后移 + G128-major W scale

v29的NCU仍有8,126,464个global excessive理论sectors；本轮检验W scale供数布局是否
能继续改善性能，不将该counter等同于实际HBM流量或直接预测加速比。

保持51/52的同一device body、64×128×128 CTA、4 warp、两/三stage、原生两路INT4、
G128独立scale和升序FP32累加，仍最后乘行A scale。新TU仅将GroupMajorScale设为true。
自然W scale输入仍是uint8[N,G]；在预分配buffer中重排为[G,N]，不修改scale值。

- prepared compute-only在计时前重排；full cold/conversion-only计入W转换，steady缓存。
- 4096³额外scale重排读写256KiB。A转换仍2个kernel，W变为3个，元数据不再声称两者相同。
- 不改变既有guard，不接受O7/O8，不回退其他数学路径。
- 53/54必须分别与51/52输出逐位相同；并继续验证相对原正式/O0的MSE和FP64语义参考。
- 不改正式默认或5090。只有完整实测后更新最佳报告。

流程：本地编译/CPU契约→GitHub→A100同步构建→旧代码保持/原生INT4审计→
数值和有限安全测试→4096配对初筛→24样本性能/MSE，必要时NCU与四模式验证。

本地预检：174项CPU测试通过；CUDA12.5独立TU编译通过。53为168寄存器、无stack/spill；
54为168寄存器、16B stack、12B spill store/load。这不是服务器CUDA12.8的资源验收结果。

## 已完成的服务器验证

源码`267362368335a4e66908a44a8a970328a51fe9d2`先推送GitHub，再以同提交bundle同步A100。
服务器CUDA12.8构建通过；binary SHA-256为
`9d7ee7d29b1b6a990bb9eefb4a700614051a44515c5dbdf8a4f32fe9a843db6a`。

- 192项GPU预检、20项入口拒绝保护通过。53/54与51/52输出逐位相同，scale重排逐元素一致。
- memcheck/synccheck各84项检查，0 errors；K768 racecheck为0 hazards。仅为有限测试。
- 142个候选实例通过同函数原生U4×S4/S4×S4及cp.async审计，没有降级为INT8。
  53为168寄存器、0 stack/local指令；54为168寄存器、16B stack、有local指令。
  宽松spill策略下均通过，不能宣称两个候选都零spill。
- 12个旧正式函数、140个旧候选的编码SASS保持不变。全SM80检查中268/272个旧函数不变，
  4个非目标mixed-binary函数发生FMUL乘数顺序/reuse提示变化；保留`passed=false`和差异文本。
  额外mixed格式回归通过，包括480项binary GEMM；不是O9/O10的新性能验收。

## 24样本 Compute-only / GEMM-only

同binary、24真实样本×5轮×5实现，共600条记录；warmup50/repeats200，循环换序。
先在每样本内汇总轮次，再对样本汇总；提升来自同样本同轮配对。
共享GPU、未锁频、保留全部离群和CV失败，bootstrap区间只作描述性比较。

|实现|Median ms|吞吐变化 / 同轮旧最佳52|95%区间|CV≥3% /120|
|---|---:|---:|---|---:|
|51，旧两阶段|0.482304|对照|—|44|
|52，旧三阶段|0.479232|基准|—|47|
|53，新两阶段|0.479232|0.00%，无明确变化|[0.99787,1.00215]|36|
|54，新三阶段|0.477952|**+0.43%**|[1.00214,1.00642]|43|

同stage对照：53相对51提升0.44%，区间[1.00225,1.00640]；54相对52提升0.43%。
同轮原正式O3为0.560640ms，54相对它的配对吞吐提升17.33%，区间[1.16810,1.17457]。
不使用上一轮0.480256ms除以本轮0.477952ms来计算优化收益。

4096³合成10轮、50条记录的初筛另行保留；它不替代上述真实trace。

## MSE

53/54与上一版行scale后移的51/52输出逐位相同，24样本MSE完全相同：

|参考|Median MSE|Mean MSE|
|---|---:|---:|
|O0|0.006653010287410|0.007578847013303|

对FP64语义参考`rtol=atol=1e-3`、对原正式MSE回归`rtol=1e-5,atol=1e-12`均通过。
相对原正式后端仍继承v29的微小FP32舍入变化；不能说与原正式逐位一致。
没有改变定点量化或各G128 scale，不涉及模型下游任务精度测试。

## NCU：访存效率改善，但不是主要计算上限

同binary、合成4096³，`--set full --cache-control none --clock-control none`，各捕获一次。
NCU时间仅作诊断；被profiling干扰的CUDA Event时间不作为性能结果。

|指标|51|53|52|54|
|---|---:|---:|---:|---:|
|NCU Duration ms|0.424320|0.421984|0.422528|0.419904|
|动态warp指令|101,941,248|102,400,000|99,622,912|99,319,808|
|Global excessive理论sectors|8,126,464|0|8,126,464|0|
|Local理论sectors|2,162,688|0|3,178,496|3,178,496|
|Eligible warps / scheduler|0.6585|0.6719|0.6390|0.6304|
|Issue active %|43.20|43.63|42.61|42.64|

两候选确实消除了W scale跨列访问不连续造成的excessive sectors；这是访存工作指标，
不是减少相同数量的HBM事务或字节。53消除spill，但并未超过旧三阶段52。
54动态指令仅减少约0.30%；IMMA/I2F/FFMA仍各16,777,216条，
shared wavefront仍29,622,272且excessive为0，寄存器仍168/线程、最多3 CTA/SM。

**小scale的合并读取有益，但主要瓶颈仍在每G128的MMA→partial合并→I2F→FP32累加
以及其调度/供数链路。** 单看访存效率或零spill会高估收益。
必要容量下界仍0.220347ms@1410MHz；理想重叠模型不是可保证达到的时间。

## 转换与端到端

完整24样本、单轮四模式，warmup50/repeats200/inner100，共288条记录。
它与五轮compute不是同一轮；不拼接median，也不以分段之和代替端到端直接计时。

|模式|52 median ms|54 median ms|配对吞吐变化 /52|95%区间|
|---|---:|---:|---:|---|
|Conversion-only|0.129828|0.133891|−2.93%|[0.96679,0.97322]|
|Compute-only|0.493056|0.491008|+1.04%|[1.00416,1.01965]|
|Cold|0.632064|0.634880|−0.32%，无明确变化|[0.99057,1.00484]|
|Steady-state|0.570624|0.565248|+0.59%|[1.00180,1.01670]|

额外256KiB scale重排在conversion-only/cold计入W转换，steady缓存；
原有A/W payload重排的32MiB/16MiB读写仍计时，不免费离线。
转换total约增加4.06微秒，说明小kernel也有成本；cold没有确认收益，steady只有小幅收益。
54四模式选定阶段CV≥3%依次为0/24、8/24、1/24、17/24，未过滤，
不等同于严格全阶段CV<3%验收。对同轮原正式，cold/steady配对吞吐提升9.42%/9.85%。

**决定：54是当前GEMM最优内部候选，不宣称cold最优；52继续保留作对照，正式默认不改。**
O7/O8未改、未重发性能数据，仍沿用v24/42。不能把O3的行scale后移推广到逐组A scale。

## 证据与下一步

`reports/o378_roof_v30/`与`runs/o378_roof_v30_*`保存环境、全部采样、MSE、审计和NCU文本。
二进制报告、完整PTX/SASS及扩展副本留在A100同名项目目录，不放进Git。
文本归档SHA-256：`24daa733a07cec0d3128c2307f74be7229b83f08cf623853ddf3433827438b7f`。
下载后先验证SHA和全部101个成员的路径/类型，再解压；新增6项原始证据一致性测试，
并将最佳报告逐数值核对改为v30后，本地180项CPU回归全部通过，无跳过。

后续优先独立测试scale供数与当前cp.async pipeline的融合，以及定点转换直接生成最终布局。
前者要测实际等待是否下降，后者必须计入全部转换成本；两者都不保证收益。
不再仅靠增大树形归约窗口或消除spill；magic-bias已停止。
跨G128改用INT32累加是另一项约束变更，未经确认不实施。

复测（先构建SM80扩展；输出目录必须不存在）：

```bash
python scripts/benchmark_a100_roof_trace.py --allow-reassociation \
  --variants o3 --output runs/o3_scale_layout_recheck \
  --samples 24 --rounds 5 --warmup 50 --repeats 200 --tunes -1 52 54

python scripts/benchmark_a100_roof_trace.py --allow-reassociation \
  --variants o3 --output runs/o3_scale_layout_four_modes \
  --samples 24 --rounds 1 --warmup 50 --repeats 200 --inner 100 \
  --all-modes --tunes -1 52 54
```

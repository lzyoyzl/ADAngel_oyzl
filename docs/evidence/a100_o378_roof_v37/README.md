# v37：L1 异步搬运缓存策略——负结果

**结论：不采用新候选。O3仍用54，O7/O8仍用59；转换最佳与正式默认、RTX5090均不变。**
这批测试回答的是：在当前tile/pipeline下，把`cp.async.cg`改为`cp.async.ca`能否缩短GEMM？答案是否定的。
按[NVIDIA PTX说明](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)，
`.cg`只在全局级缓存，`.ca`允许包括L1在内的缓存；它们不改变拷贝的数据或算术语义。

## 1. 隔离条件与真实样本结果

61复用O3/54的原device body，62复用O7/O8/59的原device body。
仍为CTA64×128×128、4warp、双原生INT4、寄存器partial、G128独立scale和FP32输出。
O3三阶段、O7/O8两阶段不变；只有16B异步拷贝的缓存策略改变。
SASS指令文本除`LDGSTS`的`BYPASS`标记外完全相同，包括顺序、寄存器与操作数。
这是指令文本检查，不是声称新旧机器码逐位相同。

24真实样本×3轮、warmup50/repeats200、同binary循环换序，保存全部648条记录。
以下为同轮配对结果，吞吐变化不是跨run的总体median相除。

|后端|原最佳/cg ms|L1候选/ca ms|配对吞吐变化|speedup描述性95%区间|
|---|---:|---:|---:|---|
|O3：54→61|0.474624|0.518144|**−8.60%**|[0.91200,0.91617]|
|O7：59→62|0.502272|0.528384|**−4.84%**|[0.94961,0.95490]|
|O8：59→62|0.504832|0.532480|**−5.40%**|[0.94455,0.95142]|

不替换原最佳；Conversion-only/Cold/Steady本批未重测，不把compute-only负结果伪装成四模式结果。
共享、未锁频GPU，CV≥3%记录没有删除。原最佳/新候选的CV失败数量分别为
O3 **36/72、37/72**，O7 **35/72、35/72**，O8 **36/72、29/72**。
因此不是严格CV全通过的验收。24样本同属一份trace，bootstrap只作描述性配对比较。

## 2. MSE与正确性

|后端|FP16参考|Median输出MSE|Mean输出MSE|缓存策略引入的变化|
|---|---|---:|---:|---|
|O3|O0|0.006653010287410|0.007578847013303|与54逐位相同|
|O7|O5|0.005536172426666|0.005053635833762|与59逐位相同|
|O8|O6|0.004411084948645|0.004381379215302|与59逐位相同|

三种参考不同，不能用这些MSE排名格式的普遍精度。O3相对更早正式实现的微小FP32重新结合差异
仍按既有策略验收；本轮61必须额外逐位等于54，并未放宽该要求。
GPU合成检查共252项；有限memcheck/synccheck各126项，均0 errors；13项异步scale保护检查通过。
完整mixed格式回归通过，包括480项binary GEMM。相关CPU测试最终238项通过；
最初三项旧源码契约检查因候选列表扩展失败，修正范围后复测，初始失败日志也保留。

## 3. NCU：数学不变，L1路径增加压力

同binary、4096³合成输入，`--set full --cache-control all --clock-control none`，预热50后仅捕获一次匹配kernel。
这里只诊断O3、O7；O8复用同一个62 kernel，但不冒充额外采集了O8 profile。
NCU会重放kernel，不能用下面的duration替代上面的正式Event延迟。

|后端/候选|NCU ms|动态warp指令|L1数据管线峰值占比|Eligible warps|Long-scoreboard / issue|
|---|---:|---:|---:|---:|---:|
|O3/54 cg|0.422976|99.32M|45.43%|0.629|0.742|
|O3/61 ca|0.462400|99.32M|58.59%|0.543|1.652|
|O7/59 cg|0.443264|116.29M|43.72%|0.774|0.169|
|O7/62 ca|0.467808|116.29M|58.46%|0.712|0.344|

两对的动态opcode计数分别完全相同，IMMA、I2F、FFMA均各16,777,216条；
168寄存器/线程和最多3个CTA/SM也不变。O3 stack16B、O7 stack8B及local访问保留，不是零spill方案。
观察到的DRAM读取量O3为27.17→28.56MB，O7为27.60→28.00MB，**没有减少**。
开启L1后，更多数据管线工作与更高的访存等待伴随更少的可发射warp，符合其延迟退化方向。
这不是“缓存越多越好”：当前tile、驻留规模与复用方式下，额外L1路径得不偿失。
不能仅凭该测试断言所有访存优化无效，也不能把stall/issue或PC采样占比当成耗时比例。

保持1410MHz的同一必要容量模型：原54/59仍约**0.220347ms**，新61/62的
profile-conditioned下界分别升为**0.253363/0.250159ms**，由L1数据管线约束主导。
这是假设资源理想重叠的必要下界，不是可达到延迟保证。本批没有接近原GEMM上界。
下一步应保留cg，针对具体供数/发射等待做局部实验，而不是继续全面启用L1。

## 4. 版本与证据

- CUDA实现：`8a4d71b888096e712fb71e627abd757302107e18`；分析脚本：`5d67cdb397a5800311a007aaa9621fcd8cb125a0`。
- binary SHA-256：`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
- 12个旧正式与148个旧候选GEMM的编码SASS未变；150个候选原生INT4/异步搬运审计通过。
  两个非目标mixed-binary函数codegen变化已披露，并完成mixed回归；全范围比较`passed=false`原样保留。
- `runs/o378_roof_v37_trace24_o3/`、`runs/o378_roof_v37_trace24_o78/`：原始Event、MSE、环境、provenance与GPU快照。
- `reports/o378_roof_v37/trace24_*_paired.json`：可重新计算的配对统计。
- `reports/o378_roof_v37/ncu_*_raw.csv`与`*_source_sass.csv`：4份完整profile的raw/source导出，另有两份分析JSON。
  `.ncu-rep`以及大型完整SASS/PTX仍在A100项目的同名`reports/`目录，没有混称为已纳入Git。
  `runs/*_ncu_*`中的Event时间受profiling干扰，**不得当作性能结果**。
- 94个文本证据经SHA和归档路径/类型检查后导入，归档SHA-256：
  `0f8a2a6da1dc5e8790a06e912f1e8e7428d5ee914c46d1be1697b02f7565d5ad`。

```bash
python scripts/benchmark_a100_roof_trace.py --output runs/cache_o3_recheck \
  --variants o3 --tunes -1 54 61 --allow-reassociation --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/benchmark_a100_roof_trace.py --output runs/cache_o78_recheck \
  --variants o7 o8 --tunes -1 59 62 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

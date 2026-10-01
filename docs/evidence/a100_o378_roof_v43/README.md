# v43：scale 寄存器复用，没有确认收益

结论：**O3 小幅退化，O7/O8 未确认收益；不采用，不切换正式默认。**
当前 GEMM 最佳仍是 O3/54、O7/O8/59，转换最佳也不变。RTX5090代码未改。

## 1. 改动与计时范围

本轮保持 CTA64×128×128、128线程、4个计算warp、64个FP32 accumulator/线程，
O3三阶段、O7/O8两阶段 cp.async.cg 流水线。两路原生U4×S4/S4×S4、独立G128 scale、
INT32重组、I2F、FP32乘法/FMA顺序及最终写回均不变。

|编号|scale读取与复用|
|---|---|
|0|当前最佳54/59的原始代码；编码SASS与最佳逐字相同|
|1|每个N片段提前加载W scale，跨两个M atom复用|
|2|每个G128提前加载本线程的整组W列scale；O7/O8同时保留A行scale，跨N片段复用|

坐标来自CuTe `partition_C(identity)`，不硬编码lane坐标；不跨G128共享scale。
与早期v1不同，本轮W scale寄存器放在M循环之外，并基于当前G128-major/片段复用实现。
源码级重复表达式不等于机器码重复读取，因此先审计，再测量，而不是预设会减少LDS。

24个真实样本×3后端×3实现×3轮，共648条compute-only记录；warmup50/repeats200，
同进程循环换序、同一原生Driver/CUDA Event计时器，预分配在计时区间外。
O7/O8从原始FP16 trace重放源格式量化，O3使用既定prepared G128数据；manifest、SHA及源格式身份保留。
只比较同run配对数据，不与之前扩展计时的绝对延迟直接相除。

## 2. 性能与MSE

|后端|控制 median ms|方案1 median ms|方案1配对吞吐变化|方案2 median ms|方案2配对吞吐变化|
|---|---:|---:|---:|---:|---:|
|O3|0.477184|0.487936|**−2.09%**|0.488448|**−2.09%**|
|O7|0.503296|0.500736|**0.00%**|0.501760|**+0.10%**|
|O8|0.504320|0.506880|**−0.41%**|0.506880|**−0.25%**|

吞吐变化是同样本同轮速度比汇总，不是两列整体median相除；两者可能方向/幅度不同。

|后端|方案1配对speedup 95% CI|方案2配对speedup 95% CI|CV≥3%：0 / 1 / 2|
|---|---|---|---:|
|O3|[0.975052, 0.983087]|[0.974895, 0.981250]|24 / 24 / 24|
|O7|[1.000000, 1.003077]|[0.997955, 1.004040]|24 / 24 / 23|
|O8|[0.993902, 1.000000]|[0.993976, 1.000000]|24 / 24 / 23|

CV每项分母72，全部保留，没有剔除离群值或重试到通过。共享、未锁频GPU；
不能仅凭时钟快照确定每个离群值的原因，也不声称所有阶段满足CV<3%。
24样本来自同一trace，有相关性，bootstrap区间作为描述性配对证据。
O7/O8区间触及或包含1，不能据小于1%的点估计宣布优化成功。
两方案不晋级conversion/cold/steady重测，本轮没有新的端到端性能结论。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|相对当前最佳|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|输出逐位一致|
|O7 / O5|0.005536172426666|0.005053635833762|输出逐位一致|
|O8 / O6|0.004411084948645|0.004381379215302|输出逐位一致|

MSE通过FP64 reduction计算；所有检查输出均为finite FP32，新旧输出MSE为0。

## 3. 指令与NCU：编译器已经做了scale复用

六个entry均包含原生U4/S4、S4/S4 IMMA和绕过L1的异步搬运，不存在INT8 MMA替代。
两个候选与控制的编码SASS不同，但资源及主要指令工作没有改善：

|后端|寄存器/线程，0/1/2|stack字节，0/1/2|spill store/load字节，0/1/2|静态指令数，0/1/2|
|---|---|---|---|---|
|O3|168 / 168 / 168|16 / 16 / 16|12 / 16 / 16（load同store）|944 / 952 / 952|
|O7/O8|168 / 168 / 168|8 / 8 / 8|8 / 8 / 8（load同store）|880 / 880 / 880|

所有版本资源上限均为3CTA/SM、12个计算warp；共享内存O3为50688B、O7/O8为34304B。
静态LDS条数O3均14、O7/O8均15。该静态数含循环体/其他用途，不是每个元素的访问次数。

NCU仅采方案2与控制，各取首个真实样本一次：`--set full --cache-control all --clock-control none`。
捕获对象绑定symbol、launch资源及cubin静态opcode指纹。O8和方案1未单独采NCU，
不能将下面O7的profile当作O8实测。NCU duration也不是上面的Event统计。

|指标|O3：0→2|O7：0→2|
|---|---:|---:|
|NCU duration ms|0.399360→0.407744|0.421888→0.422720|
|动态warp指令 M|99.319808→100.687872|116.285440→116.285440|
|动态LDS M|2.859008→2.859008|3.907584→3.907584|
|Shared wavefronts M|29.622272→29.622272|31.285248→31.285248|
|Local理论sector M|3.178496→4.194304|2.097152→2.097152|
|Eligible warps / scheduler|0.626568→0.610436|0.774803→0.776125|
|Issue active %|42.507904→41.832199|47.897550→47.798134|
|Long-scoreboard / issue|0.763029→0.808503|0.164280→0.171888|
|MIO stall / issue|0.463409→0.656343|0.542490→0.522155|

全部profile中IMMA、I2F、FFMA各16,777,216条，LDSM4,194,304条，BAR262,144条。
O7动态opcode数量逐项相同；O3动态总指令增加约1.38%，local理论sector增加约31.96%。
这些sector/wavefront是工作量计数，不是实际HBM字节；stall指标不是各因素的耗时占比。

**Insight：当前代码中的重复scale表达式，已被编译器部分合并为寄存器复用。**
把复用显式写进源码，没有减少LDS、shared wavefront或数学指令；O3反而增加spill和等待。
因此不能再把“每次输出计算都在重复从shared读scale”当作当前最佳实现的既定瓶颈。
当前仍需针对实际存在的供数/发射等待、地址计算与MMA→I2F→FMA关键路径优化，
后续应先证明机器码工作或依赖调度改善，再扩大测试，而非继续扩大scale缓存范围。

108SM、1410MHz下，固定观测工作量的理想重叠容量下界仍为**0.220347ms**，受MMA/I2F约束。
它是必要下界，不是已证实可达的时间；本轮没有向该上界取得进展，完整优化目标仍未完成。

## 4. 验证与复现

- 普通预检、memcheck、synccheck、racecheck各180项：3后端×5种shape×4种pattern×3方案；
  K128/256/384/640/4096，随机/全零/极值/零scale，非默认stream。
  新旧逐位一致，满足FP64语义参考 `rtol=1e-3, atol=1e-3`。
- memcheck/synccheck均0 errors；racecheck为0 hazards、0 errors、0 warnings。
  这是有限输入与几何检查，不是对所有尺寸/并发的普遍证明。
- 归档前本地和A100均289项相关CPU测试通过；另增4项归档复核测试，重算配对、MSE、审计与NCU。
- CUDA提交：`ec7a8a35e61fa6e08dde55e3090cfab0630acf23`；采集/解析提交：
  `f44fac3780446a9eb8948ae127e407eeee19ab9b`。本地实现、推送GitHub后才同步A100。
- 正式扩展未重建，SHA保持
  `fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

```bash
python scripts/probe_roof_scale_reuse_codegen.py --output reports/scale_reuse_recheck
python scripts/audit_roof_scale_reuse_probe.py --directory reports/scale_reuse_recheck \
  --best-sass docs/evidence/a100_o378_roof_v43/reports/o378_roof_v43/best_controls.sass \
  > reports/scale_reuse_recheck/audit.json
python scripts/validate_roof_scale_reuse_probe.py --cubins reports/scale_reuse_recheck \
  --output runs/scale_reuse_validation
python scripts/benchmark_roof_scale_reuse_probe.py --cubins reports/scale_reuse_recheck \
  --output runs/scale_reuse_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/profile_roof_scale_reuse_probe.py --directory reports/scale_reuse_recheck/ncu \
  --cubins reports/scale_reuse_recheck --runs-prefix runs/scale_reuse_ncu --candidate 2
python -m unittest discover -s tests/unit -p 'test_roof_v43_evidence.py' -v
```

文本证据、SASS、全部Event原始样本、NCU导出存于本目录。
完整cubin、PTX、Driver .so与 `.ncu-rep` 留在A100项目对应目录，不提交二进制。
传输归档SHA：`0178f9e1a22d497375a59e7fff454739d572a275b900c8449c4a00a1d80d1f28`。

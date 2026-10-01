# v42：保持4 warp，调整下一stage预取位置

**结论：两个候选均退化，不采用。GEMM最佳仍为O3/54、O7/O8/59，转换最佳及正式默认不变。**

## 1. 实验设计

隔离测试全局异步搬运与当前group计算的相对位置，不增加CTA线程数或stage数量：

|编号|下一stage的预取位置|
|---|---|
|position0|原实现：当前stage的wait/barrier之后、A片段加载之前|
|position1|当前G128的两个K64 A low/high片段加载完成后|
|position2|当前G128的第一个N64输出片段计算完成后|

三者均为CTA `64×128×128`、128线程、4计算warp（2×2），64个FP32 accumulator/线程。
O3保持3stage/50688B动态shared memory，O7/O8保持2stage/34304B。
两路原生U4×S4、S4×S4、G128顺序、scale、寄存器partial与输出代码不变。
position0的完整编码SASS与当前最佳54/59一致；候选是独立cubin，没有加入正式扩展。

移动后的copy仍写入已释放的槽位；每组只提交一次。下组消费前的wait/barrier保留，
没有在当前slot尚被读取时覆盖它。K=128且每stage只有一个G128，因此不会重复预取。
缩短copy提前量能否换来更好的寄存器生命周期/重叠，是本轮要验证的问题，不预设有收益。

|ptxas资源|position0|position1|position2|
|---|---:|---:|---:|
|寄存器 / 线程|168|168|168|
|O3 stack / spill stores / spill loads B|16 / 12 / 12|8 / 8 / 8|8 / 4 / 4|
|O7/O8 stack / spill stores / spill loads B|8 / 8 / 8|24 / 24 / 24|8 / 8 / 8|
|Driver查询最大CTA / SM|3|3|3|
|最大计算warp / SM|12|12|12|

上表spill是编译器报告的字节数，不是运行时总流量。

## 2. 24真实样本结果

24样本×3后端×3轮×3位置，共648条compute-only记录；warmup50/repeats200。
同一Driver/Event计时器，样本内循环换序，未锁频、未过滤离群值。

|后端|原控制median ms|position1 ms / 配对吞吐变化|position2 ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.478720|0.485376 / **−1.47%**|0.510976 / **−6.41%**|
|O7|0.503808|0.533504 / **−5.33%**|0.525312 / **−4.26%**|
|O8|0.506880|0.536064 / **−5.57%**|0.529920 / **−4.05%**|

配对吞吐先按同样本同轮取速度比，再在样本内汇总轮次、跨样本汇总；不是两列median直接相除。

|后端|position1速度比描述性95%区间|position2速度比描述性95%区间|CV≥3%记录数：0 / 1 / 2|
|---|---|---|---:|
|O3|[0.983087, 0.987821]|[0.932540, 0.938690]|23 / 25 / 22|
|O7|[0.942748, 0.948276]|[0.956315, 0.961089]|26 / 24 / 24|
|O8|[0.941729, 0.948077]|[0.957031, 0.962311]|25 / 24 / 24|

CV每项分母72，全部保留。区间均低于1，但这不是“所有计时阶段CV<3%”的正式验收。
两个候选均不晋级conversion/cold/steady重测，不宣称端到端收益。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|新旧输出|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位一致|
|O7 / O5|0.005536172426666|0.005053635833762|逐位一致|
|O8 / O6|0.004411084948645|0.004381379215302|逐位一致|

所有被检查输出均finite FP32。原始trace、prepared文件SHA、量化重放与source provenance均归档。

## 3. NCU：减少spill没有减少总工作

每个候选与控制单独采首个真实样本一次：`--set full --cache-control all --clock-control none`。
仅采O3/O7，不将O7 profiling当作O8实测。控制有两次独立采集，下面逐对呈现，不挑较小值。
捕获对象除symbol、线程数和寄存器外，还核对相应cubin的静态指令数及opcode分布。
NCU duration不作为上表Event延迟，也不与历史run延迟混算。

|指标|O3：0→1|O3：0→2|O7：0→1|O7：0→2|
|---|---:|---:|---:|---:|
|NCU duration ms|0.398208→0.405568|0.397920→0.433024|0.423136→0.448224|0.421696→0.441568|
|动态warp指令 M|99.32→99.58|99.32→108.49|116.29→122.18|116.29→118.72|
|Local理论sector M|3.18→2.16|3.18→1.02|2.10→6.42|2.10→2.16|
|Source shared wavefronts M|29.62→29.62|29.62→29.62|31.29→31.29|31.29→32.32|
|Eligible warps / scheduler|0.627→0.611|0.627→0.611|0.775→0.726|0.775→0.714|
|Issue active %|42.49→41.66|42.51→42.18|47.88→46.59|47.87→45.93|
|Long-scoreboard / issue|0.762→0.867|0.759→0.746|0.163→0.231|0.165→0.388|
|MIO stall / issue|0.464→0.455|0.464→0.555|0.544→0.428|0.546→0.656|

全部profile的IMMA、I2F、FFMA分别仍16,777,216条，LDSM为4,194,304条，BAR为262,144条。
没有减少数学、MMA片段加载或barrier次数。

- O3/position1：local访问减少，但eligible warp、issue active略降，long-scoreboard增加，未形成整体收益。
- O3/position2：local访问进一步减少，动态指令反而增加**9.24%**。
  主要增量包括约4.85M LOP3、1.83M IMAD、1.26M SHF；支持地址/索引重新计算和代码生成取舍的解释。
- O7/position1：动态指令增加**5.07%**，local理论sector约为控制的3.06倍；
  提前加载A后再执行copy地址计算没有降低其寄存器压力。
- O7/position2：动态指令增加**2.09%**，额外约1.05M条LDS及1.05M scale shared wavefront，
  long-scoreboard和MIO等待加重；不能仅用spill字节相近推断性能相同。

这不是“晚预取一定慢”的普遍结论，也不是各因素的独立因果消融。
源代码移动copy会同时改变编译器调度、寄存器活跃区间和重新计算；NCU能验证发生了什么，
不能仅凭stall名称把全部延迟归给一条源代码。
上述sector/wavefront是相应计数口径，不等于实际HBM流量。

基于固定观测工作量、108SM、1410MHz的理想重叠容量下界仍为**0.220347ms**，
由MMA/I2F项约束；这是必要下界，**不是已经证实可达的kernel时间**。
本轮没有接近该上界。保留原预取位置，下一步需减少真实的整数地址、scale供数工作，
而非继续移动整批copy或只追求更少spill。

## 4. 验证与复现

- 普通预检、racecheck、memcheck、synccheck各180项：3后端×5种shape×4种pattern×3位置；
  K=128/256/384/640/4096，随机/全零/极值/零scale、非默认stream。
  与54/59逐位一致，满足FP64语义参考 `rtol=1e-3, atol=1e-3`。
- racecheck为0 hazards / 0 errors / 0 warnings；memcheck/synccheck均0 errors。
  仅覆盖列出的几何与输入，不声称证明任意并发下安全。
- 六entry的同函数原生U4/S4及S4/S4审计通过，无INT8替代；控制SASS与当前最佳相同。
- 归档前A100相关CPU测试280项通过；另新增4项归档测试，重算配对、MSE、ISA、资源/安全和NCU。
- 正式扩展未重建，SHA保持
  `fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

编译、预检/安全检查及主测提交：`01b28c51a18b2759349d939b8275af65a1966a09`。
NCU及280项CPU测试提交：`99ad403cd8bb4559df00c91befda1bf218383e80`（CUDA不变）。
全部本地实现、推送GitHub后同步A100；RTX5090和正式默认未改。

```bash
python scripts/probe_roof_prefetch_codegen.py --output reports/prefetch_recheck
python scripts/audit_roof_prefetch_probe.py --directory reports/prefetch_recheck \
  --best-sass docs/evidence/a100_o378_roof_v42/reports/o378_roof_v42/best_controls.sass \
  > reports/prefetch_recheck/audit.json
python scripts/validate_roof_prefetch_probe.py --cubins reports/prefetch_recheck \
  --output runs/prefetch_validation
python scripts/benchmark_roof_prefetch_probe.py --cubins reports/prefetch_recheck \
  --output runs/prefetch_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/profile_roof_prefetch_probe.py --directory reports/prefetch_recheck/ncu_pos1 \
  --cubins reports/prefetch_recheck --runs-prefix runs/prefetch_ncu_pos1 --candidate 1
python scripts/profile_roof_prefetch_probe.py --directory reports/prefetch_recheck/ncu_pos2 \
  --cubins reports/prefetch_recheck --runs-prefix runs/prefetch_ncu_pos2 --candidate 2
python -m unittest discover -s tests/unit -p 'test_roof_v42_evidence.py' -v
```

仓库保留文本、SASS、原始Event样本、NCU导出；cubin、PTX、Driver .so和完整 `.ncu-rep`
在A100项目对应目录保存，不提交二进制。
传输归档SHA：`9ace120cc8c164817c15ba1c0b42f15138e4027e47dd2dbecdceef0a4345ab9f`。

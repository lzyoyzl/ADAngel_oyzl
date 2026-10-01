# v45：固定4096尺寸特化，未确认收益

**结论：保留GEMM最佳54/59，不改正式默认或RTX5090。**
M/N固定的O7/O8初筛微小正向，独立确认区间均触及1；完全固定M/N/K全部退化。
减少静态/动态指令不等于缩短关键路径，新增spill和搬运工作抵消了收益。

## 1. 实现

从当前G128-major的54/59出发，不叠加v44相对地址改写。

|方案|编译期尺寸|K循环|
|---|---|---|
|0|原运行时M/N/K|原最佳；编码SASS逐字匹配54/59|
|1|M=N=4096，K运行时|原结构|
|2|M=N=K=4096|显式unroll 1，避免整段K循环展开|

所有方案的原生host入口只允许4096³，其他尺寸在分配Event/launch前报错。
64位指针、CTA64×128×128、4warp、64个FP32 accumulator/线程、
O3三阶段/O78两阶段cp.async.cg、双路原生U4×S4/S4×S4、G128独立scale和数学顺序不变。
没有重建正式扩展。v20旧布局的固定尺寸测试不能替代当前G128-major布局的验证。

## 2. 24样本性能

初筛：24样本×3后端×3方案×3轮，648条compute-only记录，warmup50/repeats200。
同进程循环换序、CUDA Event，不过滤任何CV失败或离群值。
吞吐变化为同样本同轮配对比值的汇总，不是表中两列median直接相除。

|后端|控制median ms|方案1 ms / 配对吞吐变化|方案2 ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.477440|0.485120 / **−1.69%**|0.492544 / **−2.92%**|
|O7|0.504576|0.502272 / **+0.56%**|0.520192 / **−3.04%**|
|O8|0.506880|0.505856 / **+0.35%**|0.525312 / **−3.22%**|

|后端|方案1 speedup 95% CI|方案2 speedup 95% CI|CV≥3%：0/1/2，各72条|
|---|---|---|---|
|O3|[0.981172,0.985263]|[0.968750,0.971862]|24/25/22|
|O7|[1.002049,1.008247]|[0.964844,0.972387]|24/24/24|
|O8|[1.001498,1.006024]|[0.966732,0.972495]|24/25/24|

随后只对O7/O8方案1做独立24样本×5轮确认，共480条记录，仍50/200：

|后端|控制median ms|方案1 median ms|配对吞吐变化|描述性95% CI|CV失败0/1，各120条|
|---|---:|---:|---:|---|---|
|O7|0.502784|0.499712|**+0.20%**|**[1.000000,1.004090]**|58/62|
|O8|0.506624|0.503296|**+0.20%**|**[1.000000,1.004202]**|58/60|

独立确认均未达到下界大于1的标准。不反复重试到通过，不从不同run挑选最小延迟。
未锁频共享GPU，不能由快照确定每个离群值的原因，也不声称所有阶段CV<3%。
24样本来自同一trace，bootstrap区间是描述性证据，不是跨模型/输入总体保证。
本轮没有晋级conversion/cold/steady测试，没有新的端到端结论。

## 3. 输出与安全

|后端 / FP16参考|Median输出MSE|Mean输出MSE|与54/59输出|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位一致|
|O7 / O5|0.005536172426666|0.005053635833762|逐位一致|
|O8 / O6|0.004411084948645|0.004381379215302|逐位一致|

FP64计算MSE；全部输出finite FP32，新旧MSE=0。
普通预检、memcheck和synccheck各36项完整4096³检查：3后端×3方案×4种pattern
（随机/全零/极值/零scale），含非默认stream；对FP64语义参考满足rtol=atol=1e-3。
每次另有36项原生错误尺寸拒绝检查，不能用小矩阵代替固定尺寸kernel验收。

racecheck针对两个实际probe kernel（`--kernel-name kns=adangel_roof_fixed_dims_`），
覆盖3后端×3方案×随机4096³，共9项，另有36项尺寸拒绝；
0 hazards、0 errors、0 warnings。未对PyTorch/FP64参考kernel做race插桩，但仍执行参考比较。
memcheck/synccheck未使用该过滤，均0 errors。

初次未限定kernel的racecheck同时插桩参考计算，观察到高CPU/内存使用，随后主动中止。
其日志保留在`reports/o378_roof_v45/racecheck.log`，没有最终成功摘要，
**不计为通过**；通过的是`racecheck_targeted.log`及对应validation.json。
以上均为有限测试，不是完整输入空间证明。

## 4. 指令与资源

六个entry均含原生U4/S4及S4/S4 IMMA、LDGSTS.BYPASS，无INT8 MMA替代。
全部168寄存器/线程，最多3CTA/12个计算warp/SM；shared分别50688B/34304B。

|后端|静态指令：0/1/2|stack B：0/1/2|spill store/load B：0/1/2|
|---|---|---|---|
|O3|944/720/688|16/16/32|12/16/32，load同store|
|O7/O8|880/640/648|8/16/40|8/16/36，load同store|

SASS回边进一步解释了为什么静态总数下降不代表主循环明显简化：
O7/O8的G128主循环范围为方案0 `0x9e0–0x24e0`、方案1 `0x9e0–0x24f0`、
方案2 `0x9e0–0x2540`（右端不含），静态指令位置数实际为**432/433/438**。
总数880→640/648主要缩减的是循环之外的代码，而不是重复32次的核心循环。
O3对应主循环位置数367/371/363，也远没有总数944→720/688显示的降幅。
这里统计的是含分支各路径的代码位置，不是每次循环实际执行数；动态工作仍以NCU为准。

NCU分别捕获O3/O7的0对1、0对2，每项一个真实样本一次capture：
`--set full --cache-control all --clock-control none`。
symbol、launch资源及静态opcode指纹三重核对；O8没有单独profile。
NCU duration不替代正常Event配对计时。

|指标|O3：0→1|O3：0→2，独立capture|O7：0→1|O7：0→2，独立capture|
|---|---:|---:|---:|---:|
|NCU duration ms|0.399744→0.405760|0.396800→0.411360|0.423808→0.421760|0.420960→0.435488|
|动态warp指令 M|99.319808→98.549760|99.319808→96.272384|116.285440→114.565120|116.285440→115.941376|
|Shared wavefronts M|29.622272→29.622272|29.622272→29.622272|31.285248→31.285248|31.285248→37.941248|
|Local理论sectors M|3.178496→4.194304|3.178496→8.126464|2.097152→4.194304|2.097152→8.585216|
|Eligible warps/scheduler|0.626626→0.601904|0.626021→0.581066|0.774588→0.747924|0.774269→0.705534|
|Issue active %|42.489686→41.314604|42.466035→39.757344|47.874433→47.499980|47.849448→46.021387|
|Long-scoreboard/issue|0.758912→0.768985|0.763388→0.817517|0.164625→0.137981|0.166034→0.270458|
|MIO stall/issue|0.463831→0.559969|0.465167→0.722523|0.543479→0.553666|0.542586→0.480297|

MMA/I2F/FFMA各16,777,216条、LDSM4,194,304条、BAR262,144条，所有capture不变。
O7方案2新增shared工作来自LDGSTS：9.265152M→15.921152M wavefronts，
不是LDSM或数学指令次数增加；其shared excessive wavefronts为0.200704M→6.221824M。
这些是NCU的事务工作计数，不是HBM字节，也不能单凭计数认定某条源码是唯一原因。

**Insight：**固定维度减少了部分真实指令，但数学主工作没有变化，
寄存器分配/spill与搬运事务组织也随编译改变，eligible warp下降。
不能用大幅静态指令减少来预测同等速度提升。
下一步应追踪循环内具体地址/供数的活跃范围及LDGSTS事务组织，
在保持原片段复用的前提下减少真实工作，而不是继续盲目固定更多参数。
当前1410MHz、108SM、理想重叠的必要容量下界仍**0.220347ms**，
不是已证明可达到的kernel时间，本轮未确认向其靠近。

## 5. 复现与证据

源码提交：`70f108ed75e5b3ba854f35c1246c700a423db13b`，先本地实现/push再同步A100。
正式扩展SHA未变：
`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
本地318项相关CPU测试通过，含7项归档测试复算配对结果、MSE、安全范围、SASS和NCU。

```bash
python scripts/probe_roof_fixed_dims_codegen.py --output reports/fixed_dims_recheck
python scripts/audit_roof_fixed_dims_probe.py --directory reports/fixed_dims_recheck \
  --best-sass docs/evidence/a100_o378_roof_v45/reports/o378_roof_v45/best_controls.sass \
  > reports/fixed_dims_recheck/audit.json
python scripts/validate_roof_fixed_dims_probe.py --cubins reports/fixed_dims_recheck \
  --output runs/fixed_dims_validation
python scripts/benchmark_roof_fixed_dims_probe.py --cubins reports/fixed_dims_recheck \
  --output runs/fixed_dims_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/benchmark_roof_fixed_dims_probe.py --cubins reports/fixed_dims_recheck \
  --output runs/fixed_dims_confirm24 --samples 24 --rounds 5 --warmup 50 --repeats 200 \
  --variants o7 o8 --policies 0 1
python -m unittest discover -s tests/unit -p 'test_roof_v45_evidence.py' -v
```

全部原始Event样本、MSE、环境、SASS文本与NCU导出保存在此目录。
二进制cubin/PTX/Driver .so/.ncu-rep留在A100项目内。
完整文本归档SHA：
`59c7e7d716ee16f632dc9f25b67423b726cfe37dbae7ebfe1cf76b77266510be`。

# v46：G128递增行偏移，未确认收益

**保留最佳54/59，不修改正式默认或RTX5090。**
仅改payload的方案近似持平；将同一偏移同时用于scale的方案退化。
所有输出与原最佳逐位相同，MSE不变。本轮不是新的端到端结果。

## 1. 改动

每次预取原来使用`stage*M`和`stage*N`计算G128位置。
候选维护两个从0开始、每次分别增加M/N的uint32_t行偏移，避免源码中重复乘法。
预取仍按0→G-1顺序，包含2/3stage的prologue/drain。

|方案|payload|scale|
|---|---|---|
|0|原54/59|原54/59，编码SASS匹配最佳|
|1|递增行偏移|原按stage计算|
|2|递增行偏移|同样使用行偏移|

不是截断64位GPU指针，也不强制固定4096尺寸。host保留正尺寸、M64/N128/K128对齐、
M*K/N*K/M*N≤INT32_MAX及grid边界检查。
不叠加v44的完整相对地址改写或v45固定尺寸；CTA64×128×128、4计算warp、
O3三阶段/O78两阶段cp.async.cg、共享布局与同步、原生U4/S4和S4/S4、
G128独立scale、FP32累加顺序及转换全部不变。

## 2. 24样本配对结果

24样本×3后端×3方案×3轮，648条compute-only记录；
warmup50/repeats200，同进程循环换序，统一原生Driver/Event计时。
保留全部CV失败与离群值，不跨run挑最小延迟。

|后端|控制median ms|方案1 ms / 配对吞吐变化|方案2 ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.476672|0.476672 / **+0.16%**|0.558080 / **−14.43%**|
|O7|0.502784|0.502272 / **+0.10%**|0.514048 / **−2.18%**|
|O8|0.506880|0.506368 / **0.00%**|0.517632 / **−2.17%**|

|后端|方案1 speedup 95% CI|方案2 speedup 95% CI|CV≥3%：0/1/2，各72条|
|---|---|---|---|
|O3|[0.995717,1.004310]|[0.854468,0.856725]|22/25/19|
|O7|[0.995918,1.004210]|[0.974257,0.979920]|24/24/24|
|O8|[0.996939,1.006085]|[0.974409,0.982143]|25/24/24|

方案1区间均包含1，没有确认收益；方案2均明确退化，因此不晋级独立确认或四模式测试。
吞吐变化由同样本同轮配对速度比汇总，不等于两列整体median直接相除。
24样本来自同一trace，bootstrap为描述性证据；未锁频，不声称所有阶段CV<3%，
也不能由GPU快照确定每个离群值的原因。

## 3. MSE与安全检查

|后端 / FP16参考|Median输出MSE|Mean输出MSE|新旧输出MSE|
|---|---:|---:|---:|
|O3 / O0|0.006653010287410|0.007578847013303|0|
|O7 / O5|0.005536172426666|0.005053635833762|0|
|O8 / O6|0.004411084948645|0.004381379215302|0|

MSE采用FP64 reduction；所有结果finite FP32且与54/59逐位一致。
普通预检、memcheck、synccheck、racecheck**各180项**：
3后端×3方案×5形状×4种pattern（随机/零/极值/零scale），K128/256/384/640/4096，
覆盖非默认stream及不同row/column/group scale；对FP64语义参考满足rtol=atol=1e-3。
memcheck/synccheck为0 errors；racecheck为0 hazards、0 errors、0 warnings。
此预检的K4096形状是64×128×4096，不能冒充完整4096³ sanitizer覆盖；
完整4096³数值验证来自上述24真实样本。以上是有限验证，不是全输入空间证明。

## 4. 指令、资源与NCU

六entry均通过同kernel的原生U4/S4、S4/S4 IMMA和LDGSTS.BYPASS审计，无INT8 MMA替代。
全部168寄存器/线程，最大3CTA/12计算warp每SM；shared为O3 50688B、O78 34304B。

|后端|静态指令数0/1/2|stack B 0/1/2|spill store/load B 0/1/2|
|---|---|---|---|
|O3|944/936/1000|16/16/40|12/12/40，load同store|
|O7/O8|880/880/896|8/8/24|8/8/24，load同store|

NCU分别采O3/O7的0对1、0对2，每项首个真实样本一次capture：
`--set full --cache-control all --clock-control none`。
检查symbol、实际launch资源和静态opcode指纹；O8未单独profile。
NCU duration不是正常Event结果，不能据它宣布提升。

|指标|O3：0→1|O3：0→2，另一次capture|O7：0→1|O7：0→2，另一次capture|
|---|---:|---:|---:|---:|
|NCU duration ms|0.398240→0.396576|0.398240→0.472000|0.423488→0.422784|0.421728→0.432736|
|动态warp指令 M|99.319808→98.828288|99.319808→114.532352|116.285440→116.768768|116.285440→118.120448|
|Shared wavefronts M|29.622272→29.622272|29.622272→29.622272|31.285248→31.285248|31.285248→31.254528|
|Local理论sectors M|3.178496→3.178496|3.178496→16.777216|2.097152→2.097152|2.097152→6.422528|
|Eligible warps/scheduler|0.626247→0.626278|0.626909→0.605678|0.774359→0.769896|0.774681→0.754414|
|Issue active %|42.466135→42.463363|42.514069→40.904722|47.853205→47.929497|47.870370→47.187020|
|Long-scoreboard/issue|0.764742→0.739888|0.763264→0.687754|0.165043→0.146055|0.166492→0.204523|
|MIO stall/issue|0.462834→0.482522|0.464444→0.555682|0.543861→0.580507|0.544223→0.442000|

所有capture的MMA/I2F/FFMA各16,777,216条、LDSM4,194,304条、BAR262,144条，均不变。
sector/wavefront是理论访问工作，不是实际HBM字节；stall计数不是耗时百分比。

**解释与下一步：**

- 方案1没有降低shared/local工作；O3总动态指令仅少约0.49%，O7反而增加约0.42%，
  数学主工作未变，因此没有形成可确认的性能优势。
- 方案2引入持续存活的偏移状态，并伴随更多spill/local访问；
  O3总动态指令增加15.32%，O7增加1.58%，与退化相符。不能只把新增整数加法当成全部代价。
- O3方案2的long-scoreboard计数下降但运行更慢，再次说明单个stall指标变小不是成功标准。
  应同时检查实际工作量、资源分配与正常配对计时。
- 后续重点是减少真实数学/供数工作，而不是继续枚举等价地址表达式。
  跨G128整数尺度对齐累加已向用户另行询问，**尚未获准或实施**；magic-bias仍停止。

固定观测工作、108SM/1410MHz、理想重叠容量下界仍**0.220347ms**，由MMA/I2F约束。
它不是已证明可达到的kernel时间，本轮没有确认向该目标靠近。

## 5. 证据与复现

源码提交：`abf5b7825ad105a72d4de5b80abf534efb03e2bc`，先本地实现/push再同步A100。
正式扩展未重建，SHA保持
`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
325项相关CPU测试已在本地和A100通过；加入4项配对/MSE/审计/安全/NCU归档复算后，
本地329项通过。

```bash
python scripts/probe_roof_row_cursor_codegen.py --output reports/row_cursor_recheck
python scripts/audit_roof_row_cursor_probe.py --directory reports/row_cursor_recheck \
  --best-sass docs/evidence/a100_o378_roof_v46/reports/o378_roof_v46/best_controls.sass \
  > reports/row_cursor_recheck/audit.json
python scripts/validate_roof_row_cursor_probe.py --cubins reports/row_cursor_recheck \
  --output runs/row_cursor_validation
python scripts/benchmark_roof_row_cursor_probe.py --cubins reports/row_cursor_recheck \
  --output runs/row_cursor_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python -m unittest discover -s tests/unit -p 'test_roof_v46_evidence.py' -v
```

原始Event样本、MSE、环境、文本SASS及NCU导出在本目录；二进制cubin/PTX/Driver .so/
.ncu-rep保留在A100项目内。完整文本归档SHA：
`4fa8aed2c315720a08bbad421e18a7dbb31ee82dbe492a7fbd52b0fd057eba5a`。

# v44：完整相对地址计算，O3小幅正向但未确认，O7/O8退化

**不替换当前最佳54/59，不改正式默认或RTX5090代码。**
O3初筛与独立确认的配对吞吐点估计约+0.44%/+0.43%，但确认区间触及1；
O7/O8下降约4%～5%。零spill没有带来普遍性能提升，不能据此宣布接近有效吞吐上界。

## 1. 改动与约束

原代码将stage、CTA、行/列等偏移依次加到64位指针上；候选先生成完整的有界
`uint32_t`相对索引，再加到64位基址上一次。**没有将GPU地址截断为32位。**
host继续要求正尺寸、M64/N128/K128对齐、`M*K,N*K,M*N<=INT32_MAX`和合法grid；
CPU边界测试覆盖接近INT32上界的合法几何。小尺寸/GPU检查不代替完整空间安全证明。

|方案|地址计算|
|---|---|
|0|原最佳54/59；编码SASS与当前最佳逐字一致|
|1|A低/高INT4平面、W payload先形成完整相对字节索引|
|2|同1，另将O7/O8的FP32 A/W scale索引合并后再加基址|

O3没有方案2涉及的异步FP32 scale搬运，**其方案1/2编码SASS完全相同**。
预先指定O3以0对1为主比较，方案2不作为另一个候选挑选最小延迟。
CTA64×128×128、4个计算warp、64个FP32 accumulator/线程、O3三阶段/O78两阶段
cp.async.cg、两路原生U4×S4/S4×S4、K128独立scale、数学顺序、转换和最终写回均不变。
只修改搬运源地址表达式，compute body与54/59逐字一致。

## 2. 24样本配对性能

初筛：24样本×3后端×3方案×3轮，共648条compute-only记录。
warmup50/repeats200，同进程循环换序、同一原生Driver/CUDA Event计时器；
分配不在计时内，不筛掉CV失败或离群值。源数据、SHA、源格式重放和环境见各run。

|后端|控制 median ms|方案1 median ms / 配对吞吐变化|方案2 median ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.477184|0.475136 / **+0.44%**|0.475136 / +0.57%，相同SASS重复测量|
|O7|0.503296|0.530432 / **−5.46%**|0.524800 / **−4.09%**|
|O8|0.506880|0.535296 / **−5.34%**|0.529408 / **−4.16%**|

吞吐变化由同样本同轮速度比汇总，不是两列整体median直接相除。

|后端|方案1配对speedup 95% CI|方案2配对speedup 95% CI|CV≥3%：0 / 1 / 2，各72条|
|---|---|---|---:|
|O3|[1.003228,1.008565]|[1.002141,1.009657]|24 / 25 / 24|
|O7|[0.940727,0.949710]|[0.957447,0.962963]|24 / 25 / 24|
|O8|[0.944657,0.948864]|[0.955513,0.963601]|24 / 22 / 23|

随后仅对O3方案1做**独立24样本×5轮确认**，仍50/200，共240条记录，不覆盖初筛：

|后端|控制 median ms|方案1 median ms|配对吞吐变化|描述性95% CI|CV失败：0 / 1，各120条|
|---|---:|---:|---:|---|---:|
|O3|0.474624|0.472576|**+0.43%**|**[1.000000,1.008753]**|56 / 60|

点估计方向一致，但确认区间触及1，**未达到确认收益的标准**。
不通过反复重试获得“通过”，不以NCU单次duration代替Event配对证据。
未锁频共享GPU，不能由快照断言每个离群值的原因；不声称所有阶段CV<3%。
24样本来自同一trace，bootstrap是描述性证据，不表示独立模型/输入的总体置信保证。
不晋级conversion/cold/steady复测；本轮没有新的端到端结果。

## 3. 输出MSE与安全

|后端 / FP16参考|Median输出MSE|Mean输出MSE|与54/59比较|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位一致|
|O7 / O5|0.005536172426666|0.005053635833762|逐位一致|
|O8 / O6|0.004411084948645|0.004381379215302|逐位一致|

FP64 reduction；所有检查输出finite FP32，新旧输出MSE=0。
普通预检和memcheck/synccheck/racecheck各180项：3后端×5种shape×4种pattern×3方案；
K128/256/384/640/4096，随机/零/极值/零scale，非默认stream。
对FP64语义参考满足rtol=atol=1e-3，且强制逐位等于当前最佳。
memcheck/synccheck均0 errors；racecheck为0 hazards、0 errors、0 warnings。
这是有限验证；未改变量化算法，不将本轮当作重新验证所有量化格式的证据。

## 4. SASS与NCU：零spill不等于更少总工作

六个entry均审计到原生U4/S4和S4/S4 IMMA、cp.async.cg对应的LDGSTS.BYPASS，
不存在INT8 MMA替代。全部168寄存器/线程、3CTA/SM、12个计算warp；
shared为O3 50688B、O7/O8 34304B。资源上限与实际持续occupancy不是同一指标。

|后端|静态指令数：0 / 1 / 2|stack B：0 / 1 / 2|spill store/load B：0 / 1 / 2|
|---|---|---|---|
|O3|944 / 968 / 968|16 / 0 / 0|12 / 0 / 0，load同store|
|O7/O8|880 / 976 / 960|8 / 16 / 0|8 / 16 / 0，load同store|

NCU采O3的0/1、O7的0/1及另一次0/2，各为首个真实样本一次完整capture：
`--set full --cache-control all --clock-control none`。
捕获绑定symbol、实际launch资源及静态opcode指纹。O8未单独profile，不能把O7计数当作O8实测。

|指标|O3：0→1|O7：0→1|O7：0→2，独立capture|
|---|---:|---:|---:|
|NCU duration ms|0.398048→0.394944|0.420672→0.446784|0.420320→0.441120|
|动态warp指令 M|99.319808→100.958208|116.285440→130.195456|116.285440→128.876544|
|Shared wavefronts M|29.622272→29.622272|31.285248→31.285248|31.285248→31.053824|
|Local理论sectors M|3.178496→0|2.097152→4.194304|2.097152→0|
|Eligible warps / scheduler|0.626681→0.654737|0.774459→0.819059|0.774364→0.821373|
|Issue active %|42.507018→43.307931|47.853710→49.968793|47.868402→50.106556|
|Long-scoreboard / issue|0.766120→0.612894|0.166059→0.103675|0.164421→0.091316|
|MIO stall / issue|0.464784→0.638375|0.546606→0.456182|0.543982→0.486803|

全部capture中IMMA、I2F、FFMA各16,777,216条，LDSM4,194,304条，BAR262,144条。
动态LDS也未减少。sector/wavefront是理论访存工作，不是实际HBM字节；stall值不是各因素耗时占比。

**结论与下一步：**

- O3消除local访问、long-scoreboard下降，符合小幅正向趋势；但总动态指令增加约1.65%，
  MMA/I2F主工作不变，Event确认尚不足。不能单因归结为“地址指令减少”。
- O7方案1/2总动态指令分别增加约11.96%/10.83%，包括IMAD、LEA、IADD3及位操作。
  方案2虽零spill、issue active更高，却耗时更长：更多发射工作不是更多有效计算。
- 不继续用“零spill”或“源码更简洁”选优。下一候选应先证明关键路径/实际指令减少，
  保留原4warp的片段复用，避免再次用重复供数换occupancy。
  跨G128整数对齐累加尚未获得确认，不实施；也不重启magic-bias。

固定观测工作、108SM/1410MHz、理想重叠容量下界仍为**0.220347ms**，受MMA/I2F约束。
它是必要下界，不是已证明可达到的kernel时间。本轮未显著靠近它，完整目标仍未完成。

## 5. 复现与证据

源码/测试提交：`98bcecbcc7d4cab377fd8e1f35f78322a159ec6c`，本地实现推送后才同步A100。
正式扩展未重建，SHA保持
`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
本轮已有299项相关CPU测试通过，另加6项归档测试复算配对、MSE、审计、同SASS和NCU。

```bash
python scripts/probe_roof_flat_address_codegen.py --output reports/flat_address_recheck
python scripts/audit_roof_flat_address_probe.py --directory reports/flat_address_recheck \
  --best-sass docs/evidence/a100_o378_roof_v44/reports/o378_roof_v44/best_controls.sass \
  > reports/flat_address_recheck/audit.json
python scripts/validate_roof_flat_address_probe.py --cubins reports/flat_address_recheck \
  --output runs/flat_address_validation
python scripts/benchmark_roof_flat_address_probe.py --cubins reports/flat_address_recheck \
  --output runs/flat_address_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/benchmark_roof_flat_address_probe.py --cubins reports/flat_address_recheck \
  --output runs/flat_address_confirm24 --samples 24 --rounds 5 --warmup 50 --repeats 200 \
  --variants o3 --policies 0 1
python -m unittest discover -s tests/unit -p 'test_roof_v44_evidence.py' -v
```

全部原始Event样本、环境、MSE、文本SASS和NCU导出在本目录；二进制cubin/PTX/Driver .so/
`.ncu-rep`留在A100项目对应目录。完整文本归档SHA：
`89ed13f885baa6e56354ae14c6dbe556372aa0d7d713819e4e86536669c51338`。

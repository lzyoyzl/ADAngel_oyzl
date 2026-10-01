# v48：N方向寄存器流式片段宽度，没有确认收益

**保留原N64及最佳54/59，不改正式默认或RTX5090。**
N32没有降低实际寄存器分配，反而增加spill；N128没有形成预期的指令并行收益。
全部输出与原最佳逐位一致，MSE不变。本轮不是新的转换或端到端结果。

## 1. 唯一实现变量

|方案|流式N宽度|N片数|每次独立N atom|每路INT32 partial元素/线程|
|---|---:|---:|---:|---:|
|0 原54/59|64|2|4|16|
|1|32|4|2|8|
|2|128|1|8|32|

仅改变一次驻留的B片段与partial宽度。保持CTA64×128×128、2×2四warp、128线程、
每线程64个FP32输出accumulator、两路U4×S4/S4×S4、G128 scale和FP32顺序。
O3三阶段/O78两阶段cp.async.cg、shared swizzle、A预载、输入格式与转换均不变。
CuTe主机检查N32/N64/N128切片的B坐标拼接与完整B坐标一致，两K64半均覆盖。
p0完整编码SASS与原54/59相同；候选只是独立cubin，不重建正式扩展。

## 2. 24真实样本配对结果

24样本×3后端×3方案×3轮，共648条compute-only记录；50预热/200次Event测量，
同进程同输入循环换序，保留全部CV失败与原始耗时。

|后端|原N64 median ms|N32 median ms / 配对吞吐变化|N128 median ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.476672|0.513024 / **−7.10%**|0.481280 / **−1.03%**|
|O7|0.503296|0.506368 / **−0.41%**|0.523264 / **−3.61%**|
|O8|0.506880|0.508160 / **−0.40%**|0.526336 / **−3.81%**|

|后端|N32 speedup 95% CI|N128 speedup 95% CI|CV≥3%：0/1/2，各72条|
|---|---|---|---|
|O3|[0.927350,0.931864]|[0.987315,0.993644]|24/22/23|
|O7|[0.991936,1.002016]|[0.959302,0.966601]|25/27/25|
|O8|[0.991489,0.997972]|[0.960784,0.965318]|24/24/24|

N32的O7区间包含1，不能认定其稳定变慢，也没有收益证据；其余区间上界均小于1。
两候选均不晋级独立确认、端到端或默认。变化由同样本同轮配对比值汇总，
不等于直接相除表中整体median，也不能与旧run的绝对延迟相除。
共享、未锁频GPU；同trace的24样本bootstrap是描述性证据，不宣称所有CV通过，
不能从快照推断每一个离群值的确切原因。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|与原最佳输出MSE|
|---|---:|---:|---:|
|O3 / O0|0.006653010287410|0.007578847013303|0|
|O7 / O5|0.005536172426666|0.005053635833762|0|
|O8 / O6|0.004411084948645|0.004381379215302|0|

全部finite FP32、与54/59逐位相同；MSE用FP64 reduction。
预检和memcheck/synccheck/racecheck各180项：3后端×3方案×5形状×4pattern，
K128/256/384/640/4096，随机/零/极值/零scale、不同row/column/group scale及非默认stream。
对FP64语义参考满足rtol=atol=1e-3；前两sanitizer为0 errors，racecheck为0 hazards/errors/warnings。
sanitizer的K4096是64×128×4096，完整4096³数值验证来自24真实样本，不能混淆覆盖范围。

## 3. 指令资源及NCU解释

全部六entry同kernel确认两路原生INT4及LDGSTS.BYPASS，无INT8替代。
全部168寄存器/线程、最大3CTA/12计算warp每SM；shared O3 50688B、O78 34304B。

|后端|静态指令0/1/2|stack B 0/1/2|spill store/load B 0/1/2|
|---|---|---|---|
|O3|944/944/944|16/24/16|12/24/16，load同store|
|O7/O8|880/888/888|8/16/8|8/16/8，load同store|

NCU：首样本O3/O7的0对1、0对2，共8次单launch；
`--set full --cache-control all --clock-control none`，核对symbol、资源及静态opcode指纹。
没有单独profile O8。NCU duration不是正式Event延迟。

|指标|O3：0→1|O3：0→2，另次capture|O7：0→1|O7：0→2，另次capture|
|---|---:|---:|---:|---:|
|NCU duration ms|0.397376→0.434752|0.397184→0.401696|0.421312→0.423872|0.419936→0.438592|
|动态warp指令 M|99.319808→100.114432|99.319808→99.573760|116.285440→117.612544|116.285440→117.596160|
|Shared wavefronts M|29.622272→29.622272|29.622272→29.622272|31.285248→31.285248|31.285248→31.285159|
|Local理论sectors M|3.178496→6.356992|3.178496→4.194304|2.097152→4.325376|2.097152→2.162688|
|Eligible warps/scheduler|0.626925→0.564147|0.626809→0.622694|0.774438→0.775836|0.774683→0.707113|
|Issue active %|42.508446→38.871086|42.519498→42.098109|47.865618→48.254044|47.883765→45.716842|
|Long-scoreboard/issue|0.763057→0.726639|0.767697→0.829200|0.164868→0.190742|0.165419→0.165643|
|MIO stall/issue|0.465737→0.568448|0.465185→0.540338|0.543878→0.460693|0.545145→0.563644|

全部capture的MMA/I2F/FFMA各16,777,216，LDSM4,194,304，BAR262,144，均未减少。
wavefront/sector为理论访问工作，不是HBM字节；stall指标不是运行时间占比。

**Insight：**

- N32源码中的partial数组更小，却没有降低最终寄存器分配，local工作约翻倍。
  源码生存范围不能代替编译后的寄存器分配和动态访存证据。
- N128增加同时表示的独立atom，但没有减少总工作或提高驻留warp；O7 eligible与issue下降，
  “更多可并行表达的操作”并不保证机器码实际调度更好。
- 原N64在当前实现中更均衡，后续不继续单独扩大/缩小N流式宽度；应减少实际后处理/供数工作，
  或针对确切依赖改动，而非重复枚举等价的片段组织。
- 所有方案的理想重叠容量下界仍0.220347ms@108SM/1410MHz，受MMA/I2F约束。
  它不是保证可实现的延迟，本轮没有确认向该目标靠近。

跨G128整数对齐累加仍等待明确确认，未实施；magic仍停止。

## 4. 来源与复现

源码`51328135093d23d4f4df6de83f84e39424d31df6`，先本地实现/push，再A100 fetch/ffmerge。
正式扩展SHA仍`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
345项相关CPU测试本地和A100通过，另有4项归档复算测试核对性能/MSE、资源、安全、ISA、NCU。

```bash
python scripts/probe_roof_stream_width_codegen.py --output reports/stream_width_recheck
python scripts/audit_roof_stream_width_probe.py --directory reports/stream_width_recheck \
  --best-sass docs/evidence/a100_o378_roof_v48/reports/o378_roof_v48/best_controls.sass \
  > reports/stream_width_recheck/audit.json
python scripts/validate_roof_stream_width_probe.py --cubins reports/stream_width_recheck \
  --output runs/stream_width_validation
python scripts/benchmark_roof_stream_width_probe.py --cubins reports/stream_width_recheck \
  --output runs/stream_width_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python -m unittest discover -s tests/unit -p 'test_roof_v48_evidence.py' -v
```

原始Event、MSE、环境、SASS、NCU CSV和日志在此；cubin/PTX/.so/.ncu-rep及坐标检查可执行文件保留在A100项目。
文本归档SHA：`665f3f2740bf4c54d9a566d5fe757040ec9efa1576c0fe589dba5f4ffa835b00`。

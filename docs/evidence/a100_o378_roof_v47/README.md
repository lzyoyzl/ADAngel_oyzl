# v47：四个 warp 的 M/N 分工，两个候选均退化

**保留最佳54/59，不改正式默认或RTX5090。** 同一64×128×128 CTA、128线程，
将四个计算warp由2×2改为1×4或4×1。全部结果与原最佳逐位相同，但两候选都更慢。
本轮只验证GEMM候选，没有产生新的转换或端到端结果。

## 1. 实现与对照

|方案|warp M×N|每warp输出tile|每线程FP32 accumulator|逻辑片段供数字节/CTA/G128|
|---|---|---|---:|---:|
|0：原54/59|2×2|32×64|64|32768|
|1|1×4|64×32|64|40960|
|2|4×1|16×128|64|40960|

独立头文件仅修改warp几何，保留原低位payload、shared swizzle、scale布局、
两路原生U4×S4/S4×S4、逐G128 FP32顺序、O3三阶段/O78两阶段cp.async.cg与store。
逻辑供数模型为`2*64*64*WN+128*64*WM`；它不等于实际wavefront或HBM字节。
CuTe主机坐标检查：各8192输出恰好一个owner、每线程64输出、low/high坐标相同、
shared布局和大小不变。p0完整编码SASS与54/59相同。

## 2. 24个真实样本：性能与MSE

24样本×3后端×3方案×3轮，共648条；50次预热/200次Event测量。
同进程同输入循环换序，保留全部原始耗时、CV失败和离群值。

|后端|原2×2 median ms|1×4 median ms / 配对吞吐变化|4×1 median ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.478208|0.531456 / **−10.35%**|0.502784 / **−5.37%**|
|O7|0.503296|0.516096 / **−2.09%**|0.524288 / **−3.72%**|
|O8|0.506880|0.516096 / **−2.27%**|0.525312 / **−3.48%**|

|后端|1×4 speedup 95% CI|4×1 speedup 95% CI|CV≥3%：0/1/2，各72条|
|---|---|---|---|
|O3|[0.894838,0.900000]|[0.944948,0.951326]|25/23/24|
|O7|[0.975740,0.984095]|[0.960936,0.966403]|24/24/23|
|O8|[0.974104,0.982143]|[0.963035,0.966797]|24/23/24|

两方案区间上界均小于1，不晋级独立确认或四模式测试。
变化为同样本同轮配对速度比汇总，不是跨run或整体median相除。
24样本属于同一trace，bootstrap是描述性证据；共享、未锁频GPU，不声称严格CV全部通过，
也不把快照当作每个离群值的确定原因。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|与原最佳输出MSE|
|---|---:|---:|---:|
|O3 / O0|0.006653010287410|0.007578847013303|0|
|O7 / O5|0.005536172426666|0.005053635833762|0|
|O8 / O6|0.004411084948645|0.004381379215302|0|

FP64 reduction，全部finite FP32且逐位相同。预检、memcheck、synccheck、racecheck各180项：
3后端×3方案×5形状×4pattern；K128/256/384/640/4096，随机/零/极值/零scale，
不同row/column/group scale和非默认stream，对FP64语义参考满足rtol=atol=1e-3。
memcheck/synccheck为0 errors，racecheck为0 hazards/errors/warnings。
K4096 sanitizer形状是64×128×4096，不是完整4096³；后者数值证据来自24真实样本。
这些都是有限验证，不是全输入空间证明。

## 3. 资源与NCU：为什么更慢

六entry均通过同kernel原生U4/S4、S4/S4及LDGSTS.BYPASS审计，无INT8 MMA替代。
全部168寄存器/线程、3CTA/12计算warp每SM；shared为O3 50688B、O78 34304B。

|后端|静态指令0/1/2|stack B 0/1/2|spill store/load B 0/1/2|
|---|---|---|---|
|O3|944/960/936|16/24/8|12/20/4，load同store|
|O7/O8|880/888/904|8/0/16|8/0/12，load同store|

NCU分别采O3/O7的0对1、0对2，共8次首样本单launch：
`--set full --cache-control all --clock-control none`。
检查symbol、launch资源及静态指令指纹。没有单独采O8，NCU duration不是正式Event性能。

|指标|O3：0→1|O3：0→2，另次capture|O7：0→1|O7：0→2，另次capture|
|---|---:|---:|---:|---:|
|NCU duration ms|0.397664→0.456288|0.398016→0.418656|0.423456→0.433472|0.421568→0.436832|
|动态warp指令 M|99.319808→103.014400|99.319808→102.088704|116.285440→119.103488|116.285440→121.520128|
|Shared wavefronts M|29.622272→31.719424|29.622272→38.010880|31.285248→34.365946|31.285248→39.149563|
|Local理论sectors M|3.178496→5.275648|3.178496→1.081344|2.097152→0|2.097152→3.244032|
|Eligible warps/scheduler|0.626814→0.558081|0.626287→0.616671|0.774242→0.739818|0.774141→0.782517|
|Issue active %|42.517128→37.978274|42.475847→41.382763|47.866829→47.514137|47.844389→48.248930|
|Long-scoreboard/issue|0.763273→0.736957|0.760711→0.578726|0.166132→0.135416|0.164603→0.153714|
|MIO stall/issue|0.465394→0.573823|0.464360→0.642327|0.545560→0.587449|0.543594→0.521318|

所有capture的MMA/I2F/FFMA各16,777,216条、BAR262,144条不变；
LDSM均从4,194,304增至5,242,880（+25%）。sector/wavefront不是实际HBM字节，stall不是耗时占比。

**结论：** O7的1×4消除spill，O3的4×1减少spill，但都不能抵消额外供数与指令成本。
O7的4×1甚至eligible/issue略升仍更慢：处理更多工作不等于完成同一任务更快。
因此不再单纯靠改变四warp长宽比追求寄存器改善；原2×2在当前CTA上复用更均衡。

按实际动态工作重算，108SM/1410MHz理想重叠必要下界：原实现及1×4仍为0.220347ms（MMA/I2F）；
4×1变为O3 **0.231819ms**、O7 **0.239404ms**，约束转为L1TEX数据wavefront容量。
这是候选自己的上界变差，不是更接近原目标；这些下界均不是已证明可实现的延迟。
后续应减少实际供数/后处理工作，不以零spill或单个stall指标作为成功标准。
跨G128整数对齐累加仍等待用户明确确认，未实施；magic实验仍停止。

## 4. 来源与复现

源码：`eec7206dc09f6dc839bd2e95b03df9eac1e7bf39`，先本地实现/push，再A100 fetch/ffmerge。
正式扩展未重建，SHA仍为`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
335项相关CPU测试在本地和A100通过；另有归档复算测试核对全部性能/MSE、安全、ISA和NCU。

```bash
python scripts/probe_roof_warp_aspect_codegen.py --output reports/warp_aspect_recheck
python scripts/audit_roof_warp_aspect_probe.py --directory reports/warp_aspect_recheck \
  --best-sass docs/evidence/a100_o378_roof_v47/reports/o378_roof_v47/best_controls.sass \
  > reports/warp_aspect_recheck/audit.json
python scripts/validate_roof_warp_aspect_probe.py --cubins reports/warp_aspect_recheck \
  --output runs/warp_aspect_validation
python scripts/benchmark_roof_warp_aspect_probe.py --cubins reports/warp_aspect_recheck \
  --output runs/warp_aspect_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python -m unittest discover -s tests/unit -p 'test_roof_v47_evidence.py' -v
```

原始Event、MSE、环境、SASS、NCU CSV和日志在此归档；cubin/PTX/.so/.ncu-rep及坐标检查可执行文件留在A100项目。
文本归档SHA：`21f46f803279fd0a48eb2d5181304264ae88a21984964a17cc11b5dc11bc00f3`。

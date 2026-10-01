# v32：M32/N128零spill候选，实测负结果

## 结论与真实样本

**57/58不替代56。零spill、更高occupancy并没有转化为性能提升。**
以下为24真实样本×1轮，同binary循环换序，warmup50/repeats200，240条记录。
没有删离群或筛CV；speedup按同样本配对计算，非聚合median相除。

|后端|56 median ms|57 median ms|57吞吐变化 /56|58 median ms|58吞吐变化 /56|
|---|---:|---:|---:|---:|---:|
|O7|0.501248|0.537600|−6.46%|0.543744|−7.60%|
|O8|0.503808|0.539136|−6.43%|0.546304|−7.74%|

57的配对speedup描述性95%区间分别[0.92789,0.93905]、[0.92830,0.93750]；
58分别[0.92034,0.92593]、[0.92135,0.92614]。同trace样本有关联，不作为独立总体推断。
CV≥3%的记录/24，O7的56/57/58为13/10/13，O8为9/9/10，不能宣称严格CV验收。
所有候选输出与production逐位相同；新旧输出MSE为0。

|后端|FP16参考|Median MSE|Mean MSE|
|---|---|---:|---:|
|O7|O5|0.005536172426666|0.005053635833762|
|O8|O6|0.004411084948645|0.004381379215302|

合成4096³×10轮亦退化。因为GEMM未胜出，本轮没有为57/58接入四模式，
也不推测conversion/cold/steady结果；现有最佳的四模式仍使用v31独立实测。

## NCU解释

同binary full、cache-control none、clock-control none，单kernel诊断；不是正式Event计时。

|O7指标|56，M64|57，M32两阶段|58，M32三阶段|
|---|---:|---:|---:|
|NCU Duration ms|0.443456|0.472736|0.479968|
|寄存器/线程|168|128|128|
|可驻留CTA/SM|3|4|4|
|动态warp指令|118,767,616|142,901,248|142,229,504|
|LDSM指令|4,194,304|6,291,456|6,291,456|
|Shared wavefront|31,467,665|45,692,951|46,168,087|
|理论local sectors|4,063,232|0|0|
|Eligible warp/周期|0.785|1.119|1.057|
|模型必要容量下界ms@1410MHz|0.220347|0.246160|0.246801|

IMMA/I2F/FMUL/FFMA仍各16,777,216。每CTA的M减半使CTA数量加倍，重复读取W，
LDSM增加50%、总指令增加约20%、shared工作增加约47%，抵消了零spill与更高并发收益。
该实现的容量模型也因访存/指令工作增加而变差，不能继续套用56的0.220347ms作为新kernel下界。
O8同binary指标呈相同方向。wavefront/sector不是实际HBM字节，也不是耗时比例；
模型只是假设理想重叠的必要容量约束，不保证达到。

## 验证与证据范围

- CUDA12.8编译、146个候选原生INT4审计通过；57/58严格无stack/local，
  同一function包含两路原生U4×S4/S4×S4及async copy，没有INT8替代。
- 384项GPU预检；另64项M32/96边界、奇数G128数量、非默认stream、
  唯一scale模式、FP64语义参考及padded-M64逐位对照通过。
- M32 geometry的memcheck/synccheck各64项零错误，K768 racecheck零hazard；
  这是有限检查，不是对所有可能尺寸的形式证明。
- 26项guard拒绝检查和mixed全回归通过。
- 12个旧正式及144个旧候选的编码SASS不变；另4个非目标mixed-binary函数codegen变化，
  保留差异并完成mixed回归，不声称整份扩展不变。

主要证据：`runs/o378_roof_v32_trace24/results.jsonl`、`reports/o378_roof_v32/trace24_*_vs56.json`、
`ncu_o7_analysis.json`/`ncu_o8_analysis.json`及raw/source SASS CSV、audit和安全检查日志。
NCU注入过程中的Event时间不用于性能比较，保留原始数据但不当作正式结果。

源码`80ad507e2b2a4b18c835556afec9bdc4491f3c7f`。
Binary SHA256 `8fefd9b69edd574e3d6004754153e5c0d13c6499c18141f4173b13797648c0d0`。
原始文本归档`tmp/o378_roof_v32_complete.tgz` SHA256
`0f113ae5bbf591fbfaa10d684e1501387f073dfdd1ab9268018f4df0784565ae`，94个普通文件，解包前检查路径和类型。
大体积.ncu-rep及完整PTX/SASS仍保留A100项目目录，不提交GitHub。

## 实现设计

候选57/58基于v31的异步FP32 scale搬运，CTA改为`32×128×128`，
4 warp按WM1/WN4分布。保持每warp的M32，使B fragment仍跨两个M atom复用。
两/三阶段、原生U4×S4与S4×S4、G128独立scale和升序FP32 FMA不变。

目的：每线程输出accumulator从64降至32，在不强制大量spill的情况下争取4 CTA/SM。
本地CUDA12.5独立编译：57/58均128寄存器、0 stack、0 spill。
独立编译只是资源预检；上面的CUDA12.8和GPU结果才是本轮验收证据。
已接入host dispatch、setup及prepared-core实验调度，正式默认及v31对照不变。
四模式入口明确拒绝57/58，先筛选GEMM收益，胜出后再接入端到端验收，禁止误用旧M64 grid。

本地host-only CuTe检查使用实际候选Config：128线程各拥有32个输出元素，
32×128共4096个坐标均有且只有一个owner，低/高INT4 fragment坐标一致；
两/三stage Storage分别25856/38784字节，各stage的A/W scale地址均16B对齐。
检查源码：`tests/cuda/validate_roof_m32_coordinates.cu`；无GPU launch，不能替代硬件验证。

与旧v26/N64候选不同：本轮缩小M，保留N128，WM1/WN4而非WM2/WN2，
同时使用v31异步scale。不能把v26负结果直接当作本轮结果，也不能预先假定会胜出。

代价必须一起核对：当前每G128/CTA读A低高共8KiB、W8KiB；新tile读A4KiB、W8KiB，
CTA数翻倍，总payload请求量预估增加50%。M方向更细造成更多W重复消费；
缓存可能合并请求，不能据此声称HBM流量也增加50%。
数学MMA/I2F/FMUL/FMA工作不减少，LDSM和CTA管理工作可能增加。

本轮已验证host grid/元数据的M维一致、16B scale对齐、CuTe输出坐标及M32边界，
完成原生INT4审计、逐位正确性/有限安全测试、与56的同轮性能/MSE配对。
降低寄存器但总延迟增加，因此记录负结果并保留56，不追求表面零spill。
不改变INT4路径、定点格式、跨G128 FP32累加或转换计时口径。

# 当前最佳 O8：预热后 NCU 与优化查重

## 结论

**本轮没有重复优化或新增 GEMM 加速结果。** 补齐当前最佳 O8 的预热后完整 NCU，
证据仍指向计算发射/消费依赖与有限延迟隐藏，而不是首先受 DRAM 带宽限制。
暂不据此重做已失败的 tile、stage、producer warp、LDSM 替换或系数预计算。
最佳仍为 O3 v89、O7/O8 v78 GEMM + v73 准备；正式默认不变，目标尚未达到。

## 1. 为什么这次不是重复测试

已有 v90 采集的是 O3/O7 的首层样本、清缓存 kernel replay；v70 的 O8 仍是旧 v67。
本轮复用原 v78 二进制，采集真实 `layer_12_o_proj`、4096³，
采用 `--set full --replay-mode application --cache-control none --clock-control none`。
50个应用回放进程**每轮重新预热50次**，然后只采一个目标入口；不是50次配对性能轮次。

kernel replay不能恢复缓存状态；需要应用自行预热时，application replay配合不清缓存更适合。
多轮采集会重复应用设置，因此完整采集较慢。
[NVIDIA Replay/Cache Control 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control)。
未锁频警告保留。本次不与旧NCU duration直接计算加速，也不代替24样本Event结果。

## 2. 本次测得的诊断数据

|指标|当前最佳 O8，本次单样本|
|---|---:|
|NCU Duration ms，仅诊断|0.405600|
|捕获时 GPC 平均频率 GHz|1.299553|
|Eligible warp / scheduler|0.746410|
|Issue active %|46.904570|
|寄存器/线程；最大 CTA/SM|168；3|
|Achieved occupancy %|17.964700|
|动态 warp 指令 M|105.521152|
|原生 INT4 IMMA M|16.777216|
|最终 I2F M / FFMA|0.524288 / 0|
|L2 sector hit rate %|96.965764|
|DRAM读取/写入 MB（十进制）|19.872256 / 68.099712|
|LDS/LDSM excessive shared wavefront|0 / 0|
|动态 local sector|0|

|未发射 PC 采样原因|占本次未发射样本的比例|
|---|---:|
|Wait：固定延迟依赖等待|35.54%|
|Math-pipe throttle|29.11%|
|Barrier|11.74%|
|MIO throttle|9.29%|
|Short scoreboard|5.91%|
|Long scoreboard|4.39%|

共10557个未发射PC样本；**这些比例不是运行时间占比**，也不能直接把某条消费者PC上的等待归因给前一条指令。
IMMA消费者占math样本2614/3073，IMMA/IMAD消费者分别占wait样本1908/997（总wait3752）。
这支持重点关注MMA与整数加权的交错发射/依赖，但不能精确分解其耗时。
[NVIDIA Warp Stall 解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons-not-issued)。

## 3. 如何影响下一步选择

当前整数路径已把每G128的FP32转换/累加改成安全全K整数累加，最后才做I2F和base scale。
因此不能继续把旧逐组FP32 scale链当成本轮首要瓶颈；当前还存在组内整数factor加权及两路INT4消费链。
高math/wait、eligible不足1与issue约47%，说明“减少HBM字节就能接近纯MMA上限”不是当前证据支持的主要方向。

固定实际工作、1410MHz、理想重叠模型：MMA必要容量下界0.220347ms，
L1TEX data容量0.182615ms，observed DRAM/spec带宽0.056574ms。
**它们不能相加，也不是当前实现可保证达到的延迟**；本次实际频率并非固定1410MHz，
更不能将0.405600−0.220347视作可消除开销或直接预测加速。
async copy仍有excessive wavefront，不能将“shared读取无excessive”说成整条shared路径无额外工作。

|可能再次提出的方向|已有结果与本轮决策|
|---|---|
|更多stage / producer warp / private pipeline|v41/v84/v95/v109已有资源或负收益证据，不重跑|
|扩大tile、更多warp/chain|v83/v92/v104/v111已检查，不以换尺寸重新命名|
|改LDSM供数/fragment布局|v85/v86/v94/v112已有证据；本次shared读取excessive=0，不重做|
|提前算coefficients、shared coefficient表|v72/v76已有结果，不当作新方案|
|换cache策略、prefetch位置或copy分工|v37/v38/v42/v64已有结果，不重复扫描|
|重新强调逐组FP32 scale优化|当前全K路径已消除该热循环工作，不优化已不存在的瓶颈|

下一候选须能明确改变尚未测试的真实依赖/指令工作，并先说明与上述机制的区别。
仅凭此单样本报告，目前不承诺新的加速实现，也不重复已知负收益路线。

## 4. 正确性与证据范围

50/50回放：源格式identity一致、guard为2048整数CTA/0 fallback/0 invalid、finite FP32，
相对v67全K输出逐位相同（MSE=0）；本样本 O8/O6 MSE为0.00018277407059992296。
不是新的24样本MSE统计；原聚合MSE及Event性能保持原报告。
实际加载CUBIN与原v78审计SHA相同，正式扩展SHA未变；没有新CUDA编译、conversion或Cold/steady成绩。

[全部原始CSV、50份receipt与重算测试](evidence/a100_o378_roof_v114/README.md)。

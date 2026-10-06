# O3/O7/O8：优化查重与 CTA 尾部诊断（v101）

结论：**不重复已否定方向；本轮只做新的时间线诊断，没有产生新的最佳性能版本。**
当前主基准仍是 O3 v89、O7/O8 v78＋v73，v99 作为 O7/O8 微调候选保留。
观测显示，大部分执行期间已达到当前实现的 3 CTA/SM 容量；尾部确有空档，但不足以解释
与纯 MMA 必要服务时间约 0.220 ms 的主要差距。不把 Stream-K 当作当前首要优化方向。

## 1. 先查重，再决定是否测试

|方向|已有证据|本轮处理|
|---|---|---|
|CTA/warp 几何、N tile、pipeline stage|v47/v48/v57/v83/v84/v98|不重扫参数|
|producer、槽位握手、片段提前加载|v41/v42/v64/v95/v96|不重做；提前加载下一片段 B0 已在 v96 中|
|增加 MMA 链、缩短寄存器生命周期|v91/v92/v94/v96/v97|不换表达式重复同一机制|
|scale 预加载/共享、unit-factor、整数全 K|v44/v72/v74/v76/v77/v78/v79/v87|保留有效方案，不再重复已否定项|
|输入缓存、prefetch、输出 store|v37/v38/v99|v99 微收益保留，不继续相邻提示扫描|
|两路统一 S4 的精确 signed-radix16|v100|编译工作未减少，停止 GPU 性能/MSE 测试|
|当前最佳 kernel 的 CTA 执行时间线|此前没有对应时间戳证据|只做一次独立诊断，不改默认|

历史结果见 [迭代记录](o3_o7_o8_iteration_summary.md)；“编译未晋级”不等于“实测慢了”。

## 2. 诊断怎么做

固定 A100、CUDA 12.8、pinned CUTLASS、`M=N=K=4096`，取真实样本 `layer_00_q_proj`。
O3 使用 v89，O7/O8 使用 v78；转换仍用原方案，不更改量化、G128 scale 或 guard。
每种配置采集 3 次完整 kernel，共 9 份时间线；不是小矩阵性能筛选，也不是新的 24 样本验收。

每个 CTA 的四个 warp 分别由 lane0 在计算前后记录 `%globaltimer` 和 `%smid`。
CTA 的近似执行区间取四个 warp 最早 start 至最晚 end，**不是硬件分配/释放 CTA 资源的精确时刻**。
时间戳立即写入 FP32 输出末尾的 256 KiB 独立 scratch，不保存在热循环寄存器中；不增加 barrier，
不保存数值 partial，不做跨 CTA reduction。每次采集都与未插桩最佳输出逐位对照。

`%globaltimer` 的行为具有目标相关性；本次时间戳差分的最大公约数为 1024 ns，
不能声称实际计时分辨率为 1 ns，也不能直接推广到其他 GPU。
[NVIDIA PTX 12.8：globaltimer](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#special-registers-globaltimer)。

两种编译 entry 都保持 168 registers、128 threads、3 CTA/SM；整数循环仍 MMA64/LDSM16、hot local0。
原最佳 entry 完整 SASS 编码对照通过，同 entry 确认 U4/S4＋S4/S4 原生 INT4、`cp.async.cg`，无 INT8 替代。
O3 整个 entry local 仍16 B；O7/O8 插桩 entry local 从0变8 B（不在整数热循环），因此仍需承认插桩扰动。

## 3. 观测结果

下表对每种配置的 3 次采集分别计算，再取中位数。全部 2,048 个 CTA 覆盖108个 SM，最大观测并发为3 CTA/SM。

|诊断指标|O3|O7|O8|
|---|---:|---:|---:|
|时间戳覆盖跨度 ms|0.398336|0.402432|0.437248|
|达到3 CTA的 SM×时间比例|89.51%|89.72%|89.54%|
|最后一个 CTA 开始至结束 ms|0.050176|0.052224|0.056320|
|全程缺失槽位容量折算比例|5.42%|5.08%|5.25%|
|其中末尾阶段的折算比例|4.70%|4.55%|4.67%|

“缺失槽位容量折算比例”按 `∫ ΣSM (3−观测并发CTA数) dt / (108×3×观测跨度)` 计算。
例如，一段时间只有一个 CTA，而容量是三个，该段按缺两个槽位计；不是把整段时间当作损失。
这些是**槽位容量的账目，不是可达到的加速比或严格延迟上界**：CTA 占用不代表持续发射 MMA，
三个 CTA 同驻留也不代表 Tensor Core 满载；改变驻留数还会改变每个 CTA 的运行速度。

因此，不能用“7波变6.32波”直接预测约10.7%吞吐收益，也不能把上述约13%的末尾时长全部消掉。
目前证据只支持：空槽主要集中在末尾，规模约为几个百分点；kernel 主体仍需从 MMA 依赖、
就绪 warp 和非 MMA 指令预算解释。这个判断与此前 NCU 的低 eligible-warp/issue 数据相符，
但本轮没有新增 NCU 指标，也没有证明某个具体 stall 原因。

## 4. 插桩与波动的边界

未插桩同一 kernel，在采集前后各做 warmup50/repeats200；保留全部原始 Event 样本和 CV，未删离群值。

|配置|控制前 median ms / CV|插桩3次 Event ms|控制后 median ms / CV|
|---|---|---|---|
|O3|0.403456 / 5.07%|0.404480，0.398336，0.406528|0.397312 / 3.43%|
|O7|0.437248 / 4.09%|0.439296，0.408576，0.394240|0.444416 / 3.33%|
|O8|0.432128 / 4.46%|0.442368，0.443392，0.443392|0.436224 / 2.79%|

GPU 未锁频；前后快照 SM clock 从1410降至1200或1215 MHz。快照不等于每次 kernel 的实时频率，
但加上原始时延的明显波动，足以说明不能从这几次 capture 估计插桩的精确开销或优化百分比。
因此不把本表拼进正式24样本性能结果，也不为了这个诊断追加零 CV 失败的重测。

9 次采集的 FP32 输出都与对应最佳 kernel 逐位一致；下表仅是这一真实样本的输出 MSE。

|配置 / 主参考|采样输出 MSE|相对未插桩最佳的输出 MSE|
|---|---:|---:|
|O3 / O0|0.000574597343|0|
|O7 / O5|0.000171697811|0|
|O8 / O6|0.000213702315|0|

这不是新的24样本 MSE 汇总，也没有改变原有精度结论。正式扩展 SHA-256 仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`；正式默认与5090未改。

首次 O3 运行因计时前 `uint32` CUDA `any` 不受支持而停止；只将检查移到 CPU，
从新目录重跑，保留原失败日志。该失败发生在插桩 kernel 启动之前，不是数值错误。

## 5. 后续取舍与复现

当前不投入 Stream-K、persistent 或跨 CTA partial/reduction：其潜在尾部收益较有限，
还会新增归并工作并偏离此前单 CTA 完整 K、单次最终输出的约定。若将来测试，需另行确认范围。
下一候选应提出不同于历史的机制，并能在编译层面证明减少主要指令/依赖预算或增加有效并发；
仅改写已测过的 fragment、stage、cache 表达式不继续测试。最佳版本及正式结果不更新。

在已有基准 cubin 与 trace 的 A100 项目内，用新的输出目录复现：

```bash
python scripts/probe_cta_timeline_codegen.py --kind o3 \
  --output reports/o378_timeline_reproduce_o3_codegen
python scripts/probe_cta_timeline_codegen.py --kind o78 \
  --output reports/o378_timeline_reproduce_o78_codegen

python scripts/profile_cta_timeline.py --variant o3 \
  --codegen reports/o378_timeline_reproduce_o3_codegen \
  --output reports/o378_timeline_reproduce_o3
python scripts/profile_cta_timeline.py --variant o7 \
  --codegen reports/o378_timeline_reproduce_o78_codegen \
  --output reports/o378_timeline_reproduce_o7
python scripts/profile_cta_timeline.py --variant o8 \
  --codegen reports/o378_timeline_reproduce_o78_codegen \
  --output reports/o378_timeline_reproduce_o8

python -m pytest tests/unit/test_cta_timeline_probe.py \
  tests/unit/test_roof_v101_evidence.py -q
```

原始时间戳、Event 样本、审计和可重算的汇总见
[v101 evidence](evidence/a100_o378_roof_v101/README.md)。本地相关19项测试通过；
服务器使用上述命令复核。不需要重编译正式扩展或重新生成 trace。

# v88：采用 CuTe 的 MMA fragment 遍历，未确认稳定收益

**不采纳，不改变当前最佳或正式默认。** 四样本初筛略快；扩大24样本×3轮后，
O7/O8配对吞吐变化为+0.67%/+0.45%，两者95%区间均跨1。按用户要求，本轮归档后暂停任务。

## 改动与来源

参考 pinned CUTLASS 的 [CuTe gemm.hpp](https://github.com/NVIDIA/cutlass/blob/db1c288993354c88e551c40c19a8fb93a774a241/include/cute/algorithm/gemm.hpp)
中 `REGISTER .reuse OPTIMIZATIONS`：由整个fragment的 `cute::gemm` 决定M/N蛇形遍历，
替代v78手写的行优先单atom调用。只改high_K0、high_K1、low_K0、low_K1四个阶段内
独立输出atom的遍历；每个输出自身的K64/乘16/scale/跨G128累加顺序不变。

仍为八条独立链、CTA64×128×128、4warps、两stage、两路原生U4×S4/S4×S4，
同一v73准备、原full-K整数guard和FP32回退、同一最终epilogue。没有新布局或额外转换。
O3与5090未修改，也没有迁移本轮尚未确认收益的遍历策略。

|每G128整数主循环/资源|v78|v88|
|---|---:|---:|
|静态指令|383|377|
|原生S4/U4 IMMA|32 / 32|32 / 32|
|LDSM / LDGSTS|16 / 10|16 / 10|
|MMA操作数`.reuse`标记|25|27|
|寄存器/线程；最大驻留CTA/SM|168；3|168；3|
|Shared/CTA B|34304|34304|
|Stack / spill load / spill store B|0 / 0 / 0|0 / 0 / 0|
|主循环LDL / STL|0 / 0|0 / 0|

精确SASS数据流仍为16条`high→high→×16→low→low`链，每组最多8条已开始未结束。
这是程序顺序，不是硬件同时在途数。`.reuse`标记不是寄存器bank冲突或节省流量的实测。
本轮未新增NCU；不能将静态指令下降或标记增加定量解释为某项stall改善。
新cubin中的v78控制编码与原v78相同；计时直接加载原v78 cubin作为控制。

## 性能与MSE

A100，4096³，同24个真实trace；warmup50/repeats200/inner100，单stream、预分配、交错顺序。
本轮仅cached compute-only，没有新的conversion/Cold/steady结果。所有原始Event和CV失败保留。
ms为样本内跨轮median再跨样本median；speedup为同样本配对比值，不是两列总median直接相除。

|范围|后端|v78 ms|v88 ms|配对吞吐变化|speedup 95% CI|CV≥3%控制/候选|
|---|---|---:|---:|---:|---|---|
|4样本×3轮|O7|0.445440|0.443392|+1.27%|[1.011442,1.016432]|12/11，各12|
|4样本×3轮|O8|0.441600|0.437760|+0.47%|[1.002304,1.026379]|12/11，各12|
|24样本×3轮|O7|0.459776|0.457728|+0.67%，未确认|[0.997778,1.010193]|70/67，各72|
|24样本×3轮|O8|0.463616|0.458496|+0.45%，未确认|[0.999898,1.013423]|71/67，各72|

24样本有关联，区间仅作描述性比较。共享、未锁频GPU，未通过严格CV门槛；
不能用快照推断每个离群原因，也不删掉不利样本以追求正收益。

|24样本输出指标|MSE median|MSE mean|相对v78/v67|
|---|---:|---:|---|
|O7 / O5|0.005536172273439|0.005053635851003|逐位相同|
|O8 / O6|0.004411084910986|0.004381379299074|逐位相同|

48条初筛和288条确认记录全部finite FP32、metadata精确一致、输出差MSE=0。
O7全部CTA走整数路径；O8仅`layer_24_o_proj`保留12个CTA原FP32回退。原量化与安全保护未删减。

## 验证与归档

64项GPU预检涵盖两后端、四模式、随机/零/交替极值/宽scale、非默认stream、
FP64参考容差1e-3及逐位v67对照；另12项非法编码、范围、epilogue和fallback边界。
memcheck/synccheck各0 errors，racecheck 0 errors/0 warnings；每种工具完成64+12项检查。
sanitizer形状M/N≤128/256、K4096，不是4096³所有输入的内存安全证明。

首次运行在GPU启动前被审计器拒绝：静态链直方图的整数键经JSON变成字符串。
`819ac75`仅规范化JSON表示，逐项比较仍保留；编译二进制未改，失败时没有性能结果。
本地54项相关算法/证据回归通过，重算336条原始事件统计、配对区间、MSE、资源与同entry指令。
加入当前最佳v73/v78/v79的证据复核后，合计68项通过；未用本轮数字覆盖历史测量。

编译commit `866f48b`；配对驱动`2bdca75`，JSON表示修正`819ac75`。
均先本地实现并推送GitHub，再在A100 fetch/ff-only merge。
候选cubin SHA256：`db01491865d022709e528b577ff227f76c91d02b22b806058406d10413d0d3bd`。
正式扩展SHA256仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
完整下载归档`tmp/o378_v88_complete.tar.gz`两端SHA256：
`23daa4059afab37af86ddae033c6e519af9a0577a4691b85d4f15c154cf90620`。
Git保存文本证据；cubin仅保留在本地/A100及完整归档，不提交二进制。

```bash
python scripts/probe_o78_cute_traversal_codegen.py --output reports/v88_rebuild
python scripts/benchmark_o78_cute_traversal.py --cubins reports/v88_rebuild \
  --output runs/v88_recheck --samples 24 --rounds 3 --warmup 50 --repeats 200 --inner 100
```

输出目录必须不存在。上述仅用于未来明确恢复后的复现，不继续运行新任务。
结论：开源遍历实践改变了机器码，但没有证明在本负载上取得稳定收益；不继续枚举相邻遍历顺序。
保留O3 v79、O7/O8 v78+v73；原优化目标尚未达到。

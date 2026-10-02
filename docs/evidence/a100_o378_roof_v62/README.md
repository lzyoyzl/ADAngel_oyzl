# v62：O3 全 K 候选的完整四模式验证

**结论：相对同轮 O3/54＋转换2，候选 GEMM 配对吞吐 +4.02%，steady +6.22%；
转换吞吐 −7.07%，Cold 的 +2.03% 点估计尚未得到区间支持。**
这是 v61 的完整计时集成，不是新的 GEMM 指令优化。保留独立候选，不切换正式默认。

## 实现与计时

复用已审计的 O3/54、v61 GEMM cubin 和 v60 GPU guard/preparation；
原有 `roof_o3_conversion.cu` 的向量转换重新独立编译，与 v36 对应两个转换函数的编码 SASS 相同。
新增 Driver 只连接已有转换与 GEMM，未重新构建正式扩展，O7/O8 和 RTX5090 未改动。

|模式|本轮口径|
|---|---|
|conversion_only|W 转换＋候选新增 guard/factor 准备；A 转换。阶段各批量重复100次摊销，total为同索引阶段样本之和|
|compute_only|W/A/metadata 计时前准备，单次直接测 GEMM|
|cold|W 转换及 metadata 准备＋A 转换＋GEMM，单次直接测 total|
|steady_state|W/metadata 计时前缓存，只测在线 A 转换＋GEMM 的单次 total|

两路共用预分配、同一 CUDA stream、原始 INT8/MXFP4-G128 输入。
源 FP16 trace 的公共量化不计时，与已有实验一致。阶段转换仍用批量摊销；
GEMM、Cold/steady total 不用批量结果拼接。缓存是每次 benchmark 调用内复用，
**不是已实现跨 Python 调用、带权重版本管理的生产缓存**。
正常 scale/有限行 scale 预检及批次后状态读回在 Event 之外；不是主机 API wall time。
探针要求 K4096、M64/N128 对齐，unsafe 范围按 N128 CTA 回退旧算法；非法输入拒绝。

## 24 个样本配对结果

A100、4096³；24样本×1轮，50次预热、200次测量、conversion inner=100，循环交换顺序。
每个模式有24对结果，共192条；表中 ms 是样本 median 的 median。
配对 speedup 先逐样本求比值再汇总，不等于两个总体 median 直接相除。

|模式|控制 median ms|候选 median ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|Conversion-only total|0.044055|0.047370|−7.07%|[0.927159, 0.929787]|
|Compute-only GEMM|0.497152|0.475648|+4.02%|[1.023710, 1.060345]|
|Cold total|0.552960|0.537088|+2.03%|[0.994475, 1.026915]|
|Steady-state total|0.532992|0.504320|+6.22%|[1.026567, 1.078512]|

转换增加约3.315μs，包含了候选真实的 GPU 范围检查和 factor 准备，未隐藏成本。
GEMM 和 steady 的收益方向为正，Cold 区间跨1，不能宣布所有端到端模式均确认胜出。
四样本初筛也显示 GEMM +2.94%、steady +3.04%、Cold 区间跨1，原始记录另存。
v61/v62 GEMM 二进制相同；本轮百分比与 v61 不同来自配对测量环境/口径，不能当作再次提速。

### 波动说明

不锁频、共享GPU、不删除离群或失败记录。以下为 CV≥3% 条数（各24条）：

|模式|选定阶段：控制/候选|任一阶段：控制/候选|
|---|---:|---:|
|conversion_only|0 / 0|0 / 1|
|compute_only|12 / 12|12 / 12|
|cold|8 / 10|24 / 23|
|steady_state|22 / 19|22 / 19|

仍有明显段内波动/漂移，**不满足严格全阶段 CV<3% 验收**。
共享、未锁频环境是混杂因素，不能仅凭这些记录确定每个离群的原因。
bootstrap 是相关 trace 样本上的描述性比较，不能替代独立、稳定的重复实验。

## 正确性、安全与审计

|输出指标（24样本）|结果|
|---|---:|
|MSE / O0 median|0.006653010287409885|
|MSE / O0 mean|0.007578847013302748|
|候选 / 控制输出 MSE|0|

192条正式确认记录和32条初筛记录，输出均与 native O3/54 逐位一致；
packed A/W、scale 布局也逐位一致。所有真实样本 guard 状态为0。
每次预检包括96项：2种小M/N形状×6种模式数据×4种计时模式×2个实现；
另有8项非法 scale/未对齐输入拒绝。使用非默认 stream，独立 LUT/packing、FP64 语义参考。

memcheck、synccheck、racecheck 各重跑上述预检，均0 errors，racecheck另0 warnings。
范围为64×128、128×256、K4096，**不是完整4096³ sanitizer**。
候选复用 v61 同一 cubin 的两路原生 INT4、cp.async 审计；不声称本轮新增 NCU。
转换 SASS 比对通过，正式扩展 hash 不变。5项本地证据测试重算所有统计、检查完整性及审计/安全记录。

## 复现与来源

运行实现提交：`d8b4fba`，先本地提交/push，再 A100 fetch/ff-only merge。
后续证据测试补充只拒绝空汇总，不改变此次非空记录的统计规则。

```bash
PYTHONPATH=python python scripts/benchmark_roof_full_pipeline.py \
  --output runs/v62_recheck --samples 24 --rounds 1 \
  --warmup 50 --repeats 200 --inner 100
```

需要已生成的 v59 GEMM、v60 preparation 和 v61 GPU-selection cubin；
对应路径可通过脚本 `--gemm-cubins/--prep-build/--device-cubin-dir` 指定。输出目录须不存在。
本目录保留原始计时、环境、输入 hash、资源、编译/安全日志及转换 SASS；不提交二进制。

- [24样本汇总](runs/o378_roof_v62_trace24/summary.json)
- [初筛汇总](runs/o378_roof_v62_screen/summary.json)
- [转换SASS比对](reports/o378_roof_v62_conversion_compare.json)
- [复用的v61 GEMM审计](../a100_o378_roof_v61/README.md)

传输归档 SHA256：`164ef6f8b52f22812da33e5549ef2f9239ab36ad9a56c421bd00ba85512a137f`。
GEMM cubin：`d390ad87e2bfb4f3326785482cfebd2c67f36884854c904e41356e30551b9e8c`。
正式扩展：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

当前仍保留 O3/54＋转换2 为原完整对照，v61＋转换2＋guard 为新的完整独立候选；
O7/O8/59＋转换5及正式默认不变。下一步优先研究带 K 分组 scale 的成熟源码，
不继续枚举分支、tile 或 cache 写法。

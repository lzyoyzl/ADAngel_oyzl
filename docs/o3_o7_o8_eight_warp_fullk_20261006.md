# O3/O7/O8：固定八 warp 全 K 整数候选（v98）

本轮仅测试 `64×128×128`、256 线程、warp 布局 `2×4` 一个候选。
保留 O3 的 GROUP_M=8/三阶段和 O7/O8 的自然遍历/两阶段；不改变转换、量化、
各 G128 scale、整数安全 guard、原生两路 INT4 或 FP32 最终输出。正式默认、5090 不改。

最终 accumulator 从每线程64降为32，仍保留八条独立 partial 链/32个 partial 槽位，
每个 warp 不再分两次处理 N64。CTA 全局 payload/factor 复制总字节数不变。
每个 warp 的必要 MMA 64→32；CTA warp 数4→8，因此总必要 MMA **没有减少**。

## 为什么不是重复 v40

[v40](evidence/a100_o378_roof_v40/README.md) 已证实：旧逐组 FP32 路径单纯增加 warp
会因 A fragment 重复读取而变慢（O3 −4.80%，O7/O8 约 −10.7%）。
本轮只检查目前全 K 整数/合并八链是否降低足够辅助工作，不能把提高 occupancy 当作收益。
预期 CTA LDSM 指令工作从16×4增至12×8，即 **+50%**；这是风险而非优化成绩。

## 编译前固定门槛

- 每线程分配寄存器不超过128；热整数循环 local 指令不超过2。
- 同一正式候选 entry 包含原生 U4×S4、S4×S4 和 `cp.async.cg`，无 INT8 替代。
- 每 warp 热循环32条 MMA、LDSM 不超过12；按 warp 数加权后 MMA 工作不变。
- **按 warp 数加权**的静态热循环指令增量不超过10%，不能只看每线程指令减半。
- 旧对照完整编码 SASS 不变；Host CuTe 检查32/64 accumulator ownership及全部8192输出唯一覆盖。
- 门槛通过后，另查询至少2 CTA/SM、16 active warp/SM；随后直接24样本×3轮配对测试。
  warmup1000/repeats200/conversion-inner100，检查逐位输出、MSE、guard、有限安全性；不做小样本性能初筛。

编译失败/门槛不通过则停止，不启动候选 GPU、不猜测性能或新 MSE，也不扫描 warp/stage/register cap。
## 实际编译结果：不进入性能测试

|静态编译指标|O3 v89 → v98|O7/O8 v78 → v98|
|---|---:|---:|
|线程 / CTA|128 → 256|128 → 256|
|最终 accumulator / 线程|64 → 32|64 → 32|
|ptxas entry寄存器 / 线程|168 → 121|168 → 128|
|整数循环存活GPR峰值|166 → 112|166 → 113|
|整数循环每 warp 指令数|323 → 194|383 → 213|
|按 warp 数加权的整数循环指令数|1292 → 1552|1532 → 1704|
|归一化静态指令增量|**+20.12%**|**+11.23%**|
|每 CTA 热循环 MMA 指令工作|256 → 256|256 → 256|
|每 CTA 热循环 LDSM 指令工作|64 → 96|64 → 96|
|热循环 local 读写指令|0 → 0|0 → 0|

寄存器确实减少，但固定开销被更多 warp 重复执行、A fragment重复读取。
两组均超过编译前规定的10%归一化指令预算，因此停止，不调整门槛或追加邻近参数扫描。
**这不是实测负加速，也不能证明此候选一定更慢；只是没有满足本轮有限预算下的晋级条件。**
静态指令工作按warp数归一化，不是NCU动态指令计数或逐指令耗时。
ptxas寄存器不是所有硬件分配粒度的精确计费；本轮没有做Driver occupancy查询。
`.minnctapersm 2`是编译约束，不是实测驻留CTA数量。

旧对照完整编码SASS一致；候选同entry原生U4/S4、S4/S4与cg copy审计通过。
Host CuTe检查128/256线程的8192个输出唯一归属、low/high坐标和B片段边界通过。
本地/A100150项相关CPU/source/历史证据测试通过；**不是候选GPU数值正确性验收**。
没有候选kernel launch、输出/MSE、Event、NCU、四模式或sanitizer新结果。

## 当前最佳与边界（不是本轮新测）

|后端 / 独立最佳|此前24样本 GEMM median ms|MSE主参考|MSE median|MSE mean|
|---|---:|---|---:|---:|
|O3 v89|0.437760|O0|0.006653010287410|0.007578847013303|
|O7 v78 + v73准备|0.476160|O5|0.005536172273439|0.005053635851003|
|O8 v78 + v73准备|0.481280|O6|0.004411084910986|0.004381379299074|

沿用各自历史完整24样本结果，不把跨run差值当scale成本。本轮未刷新性能或MSE最佳，
也不把“保持旧最佳”报告为实测0%提升。正式扩展未重编译，SHA仍
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
接下来不继续尝试单纯增加warp；优先保留四warp的数据复用，减少实际供数/发射工作。
约0.220347ms的必要MMA容量下界仍是乐观下界，不是保证可达到的延迟；目标尚未达到。

## 证据与复现

源码先在本地实现并推送GitHub，A100再fetch bundle/fast-forward至
`ba5a927dec02b1685813581dfeee326de9808cb1`，仅在`/home/zlouyang/ADAngel_oyzl`内编译。
PTX、SASS、nvdisasm存活、生成源码、gate receipt及150项测试日志均归档，
[原始证据索引](evidence/a100_o378_roof_v98/README.md)。

```bash
python scripts/probe_eight_warp_fullk_codegen.py --kind o3 \
  --output reports/eight_warp_fullk_o3_fresh
python scripts/probe_eight_warp_fullk_codegen.py --kind o78 \
  --output reports/eight_warp_fullk_o78_fresh
python -m pytest tests/unit/test_eight_warp_fullk_probe.py \
  tests/unit/test_roof_v98_evidence.py -q
```

编译输出目录必须全新。退出0表示编译/编码/原生INT4审计完成，不代表gate通过，
更不代表性能/MSE验收；检查`codegen.json`中的`worth_runtime_validation`（本轮两组均false）。

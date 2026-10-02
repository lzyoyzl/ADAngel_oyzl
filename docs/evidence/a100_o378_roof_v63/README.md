# v63：交错 high→low 寄存器复用，资源改善但性能下降

**结论：不采纳，不扩大24样本或四模式测试。** 四个真实样本×3轮中，
O3/O7/O8 的配对 GEMM 吞吐分别为 **−6.24% / −1.64% / −1.35%**。
已测输出与当前完整对照逐位相同，MSE不变；正式默认、完整最佳及RTX5090均未改动。

## 仅测试这一项差异

对照：O3/54 和 O7/O8/59；CTA仍为64×128×128、4个warp，pipeline分别为3/2阶段。
控制cubin的编码SASS与v57固定M64控制完全相同，控制链此前已与54/59验证一致。
复用旧wrapper的符号`adangel_roof_m128_o3/o78`，**本轮两种policy的M都是64**，不是M128消融。

```text
对照：四个独立N atom，各自同时保留low和high partial
      → partial = low + 16*high → 原有逐G128 scale/FP32累加

候选：四个独立N atom交错执行，分别 high(K64_0) + high(K64_1)
      → 每个fragment乘16 → 加入两个low K64 MMA
      → 同一partial fragment → 原有逐G128 scale/FP32累加
```

每批四个N atom的源码partial由32个INT32/线程降为16个；A/B片段复用、scale索引、
FP32组顺序和最终写回不变。所有G128整数中间量在INT32范围内，乘16使用整数乘法，
不依赖对负整数左移的C++行为。没有新增量化、scale合并、magic或软件替代Tensor Core。

与历史的单atom Merge不同，本轮在四个N atom之间交错；
它仍减少了低/高两路之间的独立性，不能假定partial少就一定更快。
该候选是对当前源码的推导，不是直接移植DeepGEMM的FP8算法。

## 编译与ISA

|路径|REG/线程，前→后|ptxas spill stores/loads bytes，前→后|动态shared bytes|可驻留CTA/SM，前→后|
|---|---|---|---:|---|
|O3|168→168|12/12→0/0|50688|3→3|
|O7/O8共用kernel|168→168|8/8→4/4|34304|3→3|

两种policy同entry的PTX/SASS均有`U4×S4`和`S4×S4` INT4 MMA与`cp.async/LDGSTS`；
没有INT8 MMA替代。静态主循环仍有64条IMMA、64条I2F、64条FFMA以及16条LDSM，
没有减少必要MMA或fragment加载工作。O7/O8另保留64条FMUL。
这些是编译计数，不是本轮NCU动态计数；没有新做NCU。

## 四样本配对初筛

A100，4096³，`layer_00_{q,k,v,o}_proj`，每样本3轮，每轮warmup50/repeats200。
共72条compute-only记录，单stream、预分配、原生Driver Event，轮换控制/候选顺序。
ms为各样本跨轮median的median；speedup先做样本内配对比值再汇总，不能直接用两列median相除。
共享、未锁频GPU；没有剔除离群值或CV失败。

|后端|控制ms|候选ms|配对吞吐变化|speedup 95% CI|CV≥3%记录，控制/候选（各12条）|
|---|---:|---:|---:|---|---|
|O3|0.453120|0.482304|−6.24%|[0.933333,0.944086]|6/5|
|O7|0.486912|0.495104|−1.64%|[0.979080,0.997890]|6/5|
|O8|0.485376|0.490496|−1.35%|[0.976987,0.993750]|5/4|

这是首层四样本筛选，不是24样本正式结果。区间基于这四个样本，不能当作全部层的区间。
本轮没有conversion/Cold/steady新结果，转换代码未改，也没有新增准备成本。

|后端/主参考|MSE median，前后相同|MSE mean，前后相同|候选与控制的MSE|
|---|---:|---:|---:|
|O3/O0|0.000278282719669688|0.000283055340644016|0|
|O7/O5|0.000102067943872947|0.000099167494158983|0|
|O8/O6|0.000123493256717198|0.000122317650852208|0|

MSE变小于历史24样本汇总是因为这里仅含第一层四个样本，**不是精度改善**。
所有72条已测输出均与对应54/59逐位相同，原始FP16 trace及prepared数据hash保留在environment中。

## 正确性、安全性与判断

96项prepared-core检查：3后端×4形状×4模式×2policy。
形状为64×128×128、64×128×384、128×256×640、64×128×4096；
模式覆盖随机、零、INT8/INT6/INT4极值、不同row/column/group scale及零scale。
使用非默认stream，对FP64语义参考满足rtol/atol=1e-3，且对旧最佳逐位一致。
memcheck、synccheck各0 errors；racecheck为0 errors/0 warnings。
这是小M/N、含完整K的检查，**不是4096³ sanitizer验收或所有输入的证明**。

降低spill没有改善寄存器分配或驻留warp数量；候选还把独立的low/high累加变成依赖链。
这与变慢一致，但本轮没有NCU分离各原因的时间占比，不能精确归因多少百分比。
因此不为“零spill”继续调度扫描、不扩大该候选；保留旧实现，避免用资源指标代替实测收益。

## 复现与证据

本地实现/push后A100 fetch/ff-only merge：编译脚本`20d4c49`，测试脚本`8dea07c`。
没有重编译正式扩展，SHA保持`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/probe_roof_interleaved_merge_codegen.py --output reports/v63_rebuild
python scripts/benchmark_roof_interleaved_merge_probe.py \
  --cubins reports/v63_rebuild --output runs/v63_recheck \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
```

目录必须不存在；安全检查加`--validate-only`在相应compute-sanitizer下运行。
[原始Event与汇总](runs/o378_roof_v63_screen/summary.json)、
[编译审计](reports/o378_roof_v63/codegen.json)、SASS/PTX、生成header及全部安全日志随目录保存。
传输归档SHA256：`11273b4e374d29bb6f4e372d73d5b7a44559b4a319f17fde5eef0dd6042dfd3b`。

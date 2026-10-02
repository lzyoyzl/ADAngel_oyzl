# v64：分段异步拷贝，初筛负收益，不采纳

## 改动与来源

参考 pinned CUTLASS SM80
[`mma_multistage.h` 的 `mac_loop_iter`](https://github.com/NVIDIA/cutlass/blob/db1c288993354c88e551c40c19a8fb93a774a241/include/cutlass/gemm/threadblock/mma_multistage.h)，
把下一 stage 的异步拷贝穿插到当前 MMA 计算中，而不是全部集中发出。
仅借鉴调度思路，按本项目布局独立实现；不是复制该实现或其性能结论。

只测试一个候选，不扫描位置：

1. 当前 stage wait/barrier 后：下一组 A low/high 的前半数据与 scale。
2. 第一个 N64/M atom 的第一轮 K64 MMA 后：下一组 W 的前半数据。
3. 该 M atom 缩放累加后：下一组 A low/high 的后半数据。
4. 第二个 M atom 的第一轮 K64 MMA 后：下一组 W 的后半数据，然后 commit。

全部拷贝在第一个 N64 slice 内发完；剩余 MMA/scale 工作用于覆盖异步延迟。
prologue 保留整组拷贝，每个完整 stage 仍只 commit 一次，原 wait 与 CTA barrier 不变。
不同于 v42 的“整批 copy 整体挪动”，也不同于 v63 的 high/low partial 合并。

保持 CTA64×128×128、4warp、O3三级/O7-O8两级流水线、两路独立 INT4 partial、
G128 scale、FP32 运算顺序和最终输出不变。control 为54/59，完整编码 SASS 与 v57 M64 control 一致。
复用 `adangel_roof_m128_o3/o78` 的测试入口名，但**两种 policy 都是 M64，未测试 M128**。
不改正式扩展、正式默认或 RTX5090 实现。

## 编译与指令审计

|路径|REG/线程，前→后|spill stores/loads bytes，前→后|动态 shared bytes|CTA/SM，前→后|
|---|---|---|---:|---|
|O3|168→168|12/12→0/0|50688|3→3|
|O7/O8|168→168|8/8→0/0|34304|3→3|

同一个 entry 的 PTX/SASS 均含 U4×S4 与 S4×S4 INT4 MMA 和 cp.async/LDGSTS，没有 INT8 替代。
候选没有 LDL/STL；各 entry 的 IMMA、I2F、FFMA、LDSM 静态条数仍为64、64、64、16。

|静态代码指标|O3 控制→候选|O7/O8 控制→候选|
|---|---|---|
|总指令条数|944→976|880→936|
|LDS 指令条数|14→24|15→27|
|LDGSTS 指令条数|24→24|20→20|

这些是整个 entry 的**静态**计数，不是本轮 NCU 动态执行计数。
例如 O3 的 FMUL 包含最终 row-scale epilogue，不能当作每 G128 都乘 A scale。
本轮没有新增 NCU，不定量归因哪类 stall 导致了多少时间损失。

## 四样本配对测试

A100，4096³，`layer_00_{q,k,v,o}_proj`，3轮/样本，warmup50、repeats200。
共72条 compute-only 记录；单 stream、预分配、原生 Driver CUDA Event、交错执行顺序。
ms 为样本内跨轮 median 再跨样本 median；speedup 先计算配对比值，因此不等于两列汇总 ms 直接相除。
共享、未锁频 GPU；CV≥3%的记录全部保留，没有过滤或只选最快轮。

|后端|控制 ms|候选 ms|配对吞吐变化|speedup 95% CI|CV≥3%，控制/候选（各12条）|
|---|---:|---:|---:|---|---|
|O3|0.452608|0.481280|−5.36%|[0.933476,0.953390]|6/5|
|O7|0.488960|0.527360|−7.55%|[0.920000,0.928571]|7/5|
|O8|0.488448|0.522240|−6.97%|[0.923672,0.938976]|6/5|

这是首层四样本初筛，不是24样本正式结果；区间也只覆盖该四样本集合。
三个后端均未显示收益，停止候选，不扩大24样本或四模式测试。
**没有新的 conversion/Cold/steady 结果**；转换实现不变，未增加新预处理。

|后端/主参考|MSE median，前后相同|MSE mean，前后相同|候选与控制的 MSE|
|---|---:|---:|---:|
|O3/O0|0.000278282719669688|0.000283055340644016|0|
|O7/O5|0.000102067943872947|0.000099167494158983|0|
|O8/O6|0.000123493256717198|0.000122317650852208|0|

72条已测输出与对应54/59逐位相同。这些 MSE 小于历史24样本汇总，是首层样本选择不同，
**不是精度优化**。原始 FP16 trace 与 prepared 数据 SHA 保留于 environment。

## 正确性、安全性及结论

96项检查：3后端×4形状×4数据模式×2policy；形状为64×128×128、64×128×384、
128×256×640、64×128×4096。覆盖随机、全零、INT8/INT6/INT4极值、不同 row/column/group
scale 及零scale，非默认stream，finite FP32，FP64参考 rtol/atol=1e-3，旧最佳逐位一致。
memcheck/synccheck均0 errors；racecheck 0 errors/0 warnings。
这是小 M/N、含完整 K 的 sanitizer 范围，**不是4096³内存安全验收**。

虽然消除了 spill，但寄存器分配与驻留CTA没有改善，必要 MMA 工作也没有减少。
静态地址/控制、scale shared 读取增加，且分段发出让部分数据更晚开始搬运；这些都是候选
可能变慢的机制，未通过 NCU 分离其贡献。实测否定“套用分段copy即可更快”的假设。
因此保留54/59和v62独立全K候选，不再围绕这个分段位置做枚举。
后续仍需能减少主要数学/供数工作的新证据，而不能继续追求零spill或参考项目的源码形似。

## 复现与来源

本地提交/GitHub push 后 A100 bundle fetch/ff-only merge：`ca2ddf6a9491c01da253cc2a0445bae639fc6a83`。
未重编译正式扩展；SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/probe_roof_distributed_copy_codegen.py --output reports/v64_rebuild
python scripts/benchmark_roof_distributed_copy_probe.py \
  --cubins reports/v64_rebuild --output runs/v64_recheck \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
```

目录必须不存在；使用 `--validate-only` 可在相应 compute-sanitizer 下复核。
[原始计时与汇总](runs/o378_roof_v64_screen/summary.json)、
[编译审计](reports/o378_roof_v64/codegen.json)、生成头文件、PTX/SASS和安全检查日志均保留。
传输归档 SHA256：`b045804003e26f0b68bad0684b369e01f205972512e64bbba1ade86510c60670`。

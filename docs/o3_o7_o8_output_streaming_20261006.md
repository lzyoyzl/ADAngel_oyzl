# O3 / O7 / O8：输出 streaming-store 独立候选（v99）

## 当前状态

测试始于 2026-10-06，2026-10-07 补充验证。正式默认及 5090 未修改。
三轮完整 GEMM 配对中 O3 仅 +0.24%，区间包括无收益，不采纳。
O7/O8 为 +0.43% / +0.22%，独立四模式也确认 GEMM/Cold 微收益，保留为正向微调候选。
v78 主基准继续保留，O7 steady 收益未确认；不声称所有模式均有收益或主要瓶颈已解决。
本轮只改安全全 K 整数路径的最终输出缓存提示，不扫描 tile、stage 或缓存参数。

## 与已有迭代的区别

v37 改输入 `cp.async.cg→ca`，v38 改输入 L2 预取提示，两者均已测试，不重做。
v89 改 CTA 遍历。v99 保留各自最佳遍历，仅把原输出 store 改为 CUDA `__stcs`。

v90 现有 source NCU 显示 O3 v89 / O7 v78 的 4,194,304 条动态 LDSM 均没有 excessive
shared wavefronts；因此不再把 payload bank conflict 当作尚未修复的供数瓶颈。
当前输出是普通 STG。4096² FP32 输出为 64 MiB；它可能挤占跨 CTA/跨调用复用的输入。
这是待验证的缓存假设，不是已证明的耗时原因，更不代表能突破 MMA 的必要容量下界。

[`__stcs` 官方接口](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#store-functions-using-cache-hints)
支持 float / float2。[PTX cache operators](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#cache-operators)
说明 `st.cs` 以 evict-first 限制 streaming 输出污染，缓存提示不改变内存一致性语义。

## 不变项与预先验收口径

- G128 原 scale、量化、两路原生 U4/S4 与 S4/S4、FP32 输出、partial 算术均不变。
- CTA 64×128×128、128 线程、32 partial 槽位 / 8 链；O3 三 stage，O7/O8 两 stage。
- 输入 cg、guard、fallback、转换和计时不改；fallback 仍使用原普通输出 store。
- 同 cubin 的旧最佳对照必须与原二进制编码一致。
- 先确认候选 PTX/SASS streaming store；R≤168，主循环 liveness/指令/local 不增加，
  仍 32+32 MMA / 16 LDSM，输出 store 数量不增加，再查询至少 3 CTA/SM 的资源容量。
- 编译筛选通过后直接 24 样本×3 轮配对；warmup=1000、repeats=200、inner=100。
  保留所有 CV/离群值，检查输出逐位相同及 MSE；不做小样本性能初筛。
- 只有确认净收益才扩展 conversion/Cold/steady 与有限安全检查。不把缓存命中率或
  静态指令改善直接称为加速；负结果也记录并停止，不扫描更多 store policy。

## 实现与审计结果

局部生成候选 header，只替换整数 epilogue 的三处 `float2/float` store；
CPU 测试将它们恢复后与旧完整 header 比较，确认其他数学表达式不变。
guard 和不满足安全整数累加条件时的 FP32 fallback 都保留；fallback store 不改。
CUDA Driver 查询两组均为 168 寄存器、128 线程、最大 3 CTA/SM。

|整数热循环 / 编译指标|O3 v89 → v99|O7/O8 v78 → v99|
|---|---:|---:|
|静态指令 / warp / G128|323 → 323|383 → 383|
|最大活跃 GPR|166 → 166|166 → 164|
|原生 U4/S4 + S4/S4 MMA|32+32 → 不变|32+32 → 不变|
|LDSM|16 → 16|16 → 16|
|热循环 local 读写|0 → 0|0 → 0|
|完整 entry 静态 STG / 其中 streaming|192 / 0 → 192 / 96|192 / 0 → 192 / 96|

同 cubin 的旧对照与旧最佳编码完全一致；正式候选同 entry 确认 `cp.async.cg`、
两路原生 INT4 PTX/SASS 和 `st.global.cs` / SASS `STG.*.EF`。
O3 整个 entry 仍有 local 空间（16→8B），不能把“整数热循环零 local”说成整个函数无 spill。
O7/O8 entry local 为 0。共享内存 O3 为 50,688B，O7/O8 为 34,304B，均未变。

**源码只改 store 提示，但机器码不只是缓存位不同**：编译器也调整了部分辅助指令/寄存器分配。
本轮无新 NCU，不能将全部微小收益归因于降低缓存污染，更不能宣称已解决主要瓶颈。
MMA 工作量不变，原必要容量下界约 0.220347ms（1410MHz、理想重叠）也不变。

## 完整 24 样本 GEMM 配对结果

同进程对照旧最佳与候选，三轮、每轮 200 次直接 Event 计时，warmup=1000。
O3 共 144 条、O7/O8 共 288 条记录；所有数据均为 4096³ 真实样本。
每个样本先对三轮 latency 取中位数，再汇总 24 样本。
配对吞吐先计算同样本同轮的旧/新比值，再汇总，不直接除下表两列总体中位数。

|case|旧最佳同轮 median ms|v99 median ms|配对吞吐提升|配对 speedup 95% CI|CV≥3% 旧 / 新|
|---|---:|---:|---:|---|---:|
|O3|0.438784|0.434176|+0.24%|[1.000000, 1.011792]|12/72 / 14/72|
|O7|0.474368|0.473088|+0.43%|[1.002222, 1.004334]|0/72 / 0/72|
|O8|0.475136|0.478208|+0.22%|[1.002137, 1.003219]|2/72 / 1/72|

O8 总体 latency 中位数与配对比值方向不同，原因是两种聚合不等价；不挑选更好看的口径。
所有 CV/离群值原样保留，不删时间序列，也不以复测覆盖首测。

## 正确性与 MSE

432 条真实记录均与旧最佳输出逐位一致，payload/scale/guard 一致，输出有限 FP32。
MSE 按 FP64 reduction，参考仍为 O3→O0、O7→O5、O8→O6，不是与原始模型最终 logits 比较。

|case|参考|旧 / 新 median MSE|旧 / 新 mean MSE|
|---|---|---:|---:|
|O3|O0|0.00665301028741|0.00757884701330|
|O7|O5|0.00553617227344|0.00505363585100|
|O8|O6|0.00441108491099|0.00438137929907|

补充 GPU 数值检查覆盖全编码、随机、零、极值、非默认 stream、guard 回退及 tail：
O3 96 项数值/模式检查、8 项非法输入拒绝、2 项坐标/回退检查；
O7/O8 64 项数值/模式检查、12 项边界检查、8 项坐标/混合回退检查均通过。
这些小 M/N、完整 K4096 检查用于正确性，不是小样本性能筛选或 4096³ sanitizer。

## 独立四模式确认

另外运行完整 24 样本 × 1 轮，共 384 条记录；不覆盖上面的三轮测试。
以下各表使用同一次确认 run，不与历史最快值拼接。
转换阶段 inner=100 摊销；GEMM、Cold/steady total 使用单次直接 CUDA Event。
原始 FP16→源格式的公共准备仍在实验转换计时之外，所有显存提前分配。

### Conversion-only：权重 + 激活转换

|case|旧 ms|v99 ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|O7|0.066959|0.066908|+0.03%|[0.998554, 1.001532]|
|O8|0.091392|0.091284|−0.02%|[0.998262, 1.000224]|

转换源码与输入不变，两组区间都包括无变化；没有新的 conversion 优化收益。

### Compute-only：已转换输入的 GEMM

|case|旧 ms|v99 ms|配对吞吐提升|speedup 95% CI|
|---|---:|---:|---:|---|
|O7|0.480768|0.479232|+0.21%|[1.002132, 1.004264]|
|O8|0.481280|0.479744|+0.43%|[1.002252, 1.004273]|

第二次完整配对也支持微小 GEMM 收益，但幅度远不足以解决距离容量下界的差距。

### Cold：权重、激活在线转换 + GEMM

|case|旧 total ms|v99 total ms|配对吞吐提升|speedup 95% CI|
|---|---:|---:|---:|---|
|O7|0.557056|0.556032|+0.19%|[1.001842, 1.003683]|
|O8|0.580608|0.579584|+0.18%|[1.000883, 1.003540]|

### Steady-state：缓存权重，只保留在线激活处理 + GEMM

|case|旧 total ms|v99 total ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|O7|0.528384|0.527872|+0.19%（未确认）|[0.992337, 1.002885]|
|O8|0.544768|0.543744|+0.19%|[1.001883, 1.003766]|

Cold/steady total 不是把批量转换中位数与 GEMM 中位数相加。
确认 run 任一阶段 CV≥3% 的记录，O7 为旧/新 1/96、6/96，O8 为 2/96、3/96；
各表选定阶段 CV 失败分别 O7 0/96、2/96，O8 2/96、3/96。保留全部记录，不声称零 CV 失败。
新增 384 条输出同样逐位一致、MSE 不变；总计 816 条真实记录通过本轮输出对照。

## 有限安全检查与采用决定

O7/O8 独立执行 `compute-sanitizer --tool memcheck` 的 validate-only 全部检查，
报告 **0 errors**。范围是小 M/N、K4096、非默认 stream、tail 和混合 guard/fallback；
不是完整 4096³ sanitizer，也未做本轮 racecheck/synccheck，O3 本轮未做 sanitizer。

- O3 不采纳输出提示：三轮区间包括 1，不扩大四模式或提示扫描。
- O7/O8 保留 v99 微调候选：两次全 24 样本确认 GEMM 微收益，Cold 也有小正收益。
  O7 steady 未确认，原 v78 主基准保留；正式默认及 5090 不变。
- 目标未达到。该方向到此收口，不重复输入缓存、相邻 store hint 或更多 tile/stage 扫描。
  不把 0.2%～0.4% 提升包装成主要 scale/MMA 瓶颈已突破。

## 复现与证据

源码提交 `90dfbf7496919da1c34d095354eb51e982beb219`，运行接口提交
`c74a7465f82829fbef437f2dbce9a5c348775e23`；均本地实现 → GitHub push → A100 项目内同步。
正式 `_sm80.so` 未重建。独立 cubin 和预分配、单 CUDA stream、双轨计时均有身份记录。

```bash
python scripts/probe_output_streaming_codegen.py --kind o3 --output reports/o378_roof_v99_o3_codegen
python scripts/probe_output_streaming_codegen.py --kind o78 --output reports/o378_roof_v99_o78_codegen

python scripts/benchmark_o378_output_streaming.py --kind o3 \
  --codegen reports/o378_roof_v99_o3_codegen --output runs/o378_roof_v99_o3 \
  --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python scripts/benchmark_o378_output_streaming.py --kind o78 \
  --gpu-build reports/o378_roof_v73_codegen --baseline reports/o378_roof_v67_codegen \
  --cubins reports/o378_roof_v99_o78_codegen --output runs/o378_roof_v99_o78 \
  --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100

# 独立四模式确认：不能覆盖上面两次 run。
python scripts/benchmark_o378_output_streaming.py --kind o78 \
  --gpu-build reports/o378_roof_v73_codegen --baseline reports/o378_roof_v67_codegen \
  --cubins reports/o378_roof_v99_o78_codegen --output runs/o378_roof_v99_o78_four \
  --samples 24 --rounds 1 --warmup 1000 --repeats 200 --inner 100 --full-modes

compute-sanitizer --tool memcheck --error-exitcode 86 \
  python scripts/benchmark_o378_output_streaming.py --kind o78 \
  --gpu-build reports/o378_roof_v73_codegen --baseline reports/o378_roof_v67_codegen \
  --cubins reports/o378_roof_v99_o78_codegen --output runs/o378_roof_v99_o78_memcheck \
  --samples 24 --rounds 1 --warmup 0 --repeats 1 --inner 2 --validate-only

python scripts/analyze_output_streaming.py \
  --o3 docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o3 \
  --o78 docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78 \
  --four docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78_four \
  --output reports/o378_v99_reanalysis.json
```

所有输出路径必须是新目录。完整文本、raw timings、SHA receipts 和归档信息见
[v99 evidence](evidence/a100_o378_roof_v99/README.md)。不扫描相邻 store/cache 策略，
也不重复 v37/v38 的输入缓存测试；候选是否晋级取决于实际确认，不取决于源码看起来更先进。

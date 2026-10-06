# O3 / O7 / O8：N64 边界的软件流水线候选（v96）

结论：**A100 编译完成，但预期跨 N64 边界的交错没有落到机器码，编译 gate 失败。**
不启动 GPU 性能/MSE 测试，不采纳，不切换正式默认或 RTX 5090；当前最佳不变。

## 改动及依据

对当前最佳机器码追踪发现，两个 N64 片段各有 8 条 high/high/low/low MMA 链。
编译器已重叠部分 scale 后处理，不能声称旧实现是完全串行；但第二片段的第一条
MMA 都排在第 33 条，即第一片段全部 32 条 MMA 之后。两个边界的已启动未完成链计数降到 0。
这是静态依赖顺序，不是实际并发数或量化的 stall 时间。

候选保持 32 个 partial 寄存器槽位：第一片段某条链完成最后一次 low MMA 后，先将
四个结果乘原 factor 并加入原 accumulator，再清空同一槽位，立即启动第二片段对应
high MMA。不等待第一片段其他七条链全部完成，尝试重叠收尾和下一片段启动。

B 的 K64 前半片段只在所有旧 low0 使用后覆盖，后半片段只在所有旧 low1 使用后覆盖。
A 保持原复用；不增加必要 MMA、LDSM、global payload、shared buffer 或同步。
CTA64×128×128、128 线程、O3 三阶段/分组 M8、O7/O8 两阶段，以及所有 G128 factor、
全 K INT32 安全 guard、FP32 回退与最终输出顺序均不变。

## 编译与审计结果

|整数热循环指标|O3 v89 → v96|O7/O8 v78 → v96|
|---|---:|---:|
|第二 N64 第一条 MMA 的组内序号|33 → 33|33 → 33|
|第二片段在旧八条 MMA 链结束前启动|否 → 否|否 → 否|
|已启动未完成 MMA 链峰值|8 → 8|8 → 8|
|每 G128 MMA / LDSM|64 / 16 → 不变|64 / 16 → 不变|
|整数循环静态指令|323 → 323|383 → 384|
|整数循环静态活跃 GPR 峰值|166 → 166|166 → 160|
|entry 分配寄存器 / 线程|168 → 168|168 → 168|
|整数循环 local 读写指令|0 → 0|0 → 0|

同 entry 两路原生 INT4 和 cg copy、旧对照的编码一致审计均通过。
候选本身与旧 entry 的完整编码不同，因此不声称整个 kernel 被优化成逐位相同的机器码；
但目标边界排序未改变。O7/O8 的活跃范围缩短没有降低分配寄存器，且多了一条 NOP。
没有新 CUDA 占用容量查询、kernel launch、Event、输出/MSE、NCU 或 sanitizer 结果。
预设排序 gate 失败，停止该候选，而不是试图从静态小变化推测实测加速或退化。

本地 117 项相关测试通过；A100 首批 17 项源码/代数测试通过，最终同步后再复核同一测试集。
完整原始编译结果保留于 A100 和本地 `tmp/o378_v96_compile_complete.tar.gz`，两端 SHA256：
`d357cc74c3f57e061074dc957857e54a46db12d50c51082c6fc9bf45eabc3d16`。
[编译日志、PTX/SASS、数据流及等价检查](evidence/a100_o378_roof_v96/README.md)。

## 预设验收口径

先验证旧 kernel 编码相同、同 entry 原生 U4/S4 + S4/S4 MMA 和 cp.async.cg，
再检查真实 SASS 确实在旧八条链全部结束前启动第二片段，同时峰值不超过八条链。
要求保持 64 MMA / 16 LDSM、分配寄存器不超过 168、热循环 local 指令不超过 2，
运行时三 CTA/SM 容量不降低。编译 gate 不通过则不投入 GPU 性能测试。

通过后直接进行完整 24 样本、三轮、CUDA Event 配对 GEMM；不做小样本性能初筛。
转换使用完全相同的 O3 conversion2 / O7/O8 v73，并验证有限 FP32、逐位输出、
相对 O0/O5/O6 的 MSE、guard 与各边界模式。保留全部离群值和 CV 失败记录。
只有已确认收益才扩展四模式、有限 sanitizer 与必要 NCU；静态排序改善不是加速结果。

## 复现入口

```bash
python scripts/probe_interleaved_tail_codegen.py --kind o3 --output reports/o378_roof_v96_o3_codegen
python scripts/probe_interleaved_tail_codegen.py --kind o78 --output reports/o378_roof_v96_o78_codegen

# 本次 gate 为 false，下面两个驱动会明确拒绝，不应强行运行。
python scripts/benchmark_o3_interleaved_tail.py --output runs/o378_roof_v96_o3 \
  --warmup 1000 --repeats 200 --rounds 3 --inner 100
python scripts/benchmark_o78_interleaved_tail.py --cubins reports/o378_roof_v96_o78_codegen \
  --output runs/o378_roof_v96_o78 --warmup 1000 --repeats 200 --rounds 3 --inner 100
```

代码先在本地实现并推送 GitHub，再同步到 A100；不修改 `/home/zlouyang` 以外内容。
当前已确认最佳（此前各自完整 24 样本测试，不是本轮新测）：

|后端|GEMM median ms|MSE 主参考|MSE median|MSE mean|
|---|---:|---|---:|---:|
|O3 v89|0.437760|O0|0.006653010287410|0.007578847013303|
|O7 v78 + v73|0.476160|O5|0.005536172273439|0.005053635851003|
|O8 v78 + v73|0.481280|O6|0.004411084910986|0.004381379299074|

不同 run 不拼接为新的配对收益。目标仍未达到；约 0.220347 ms 的必要容量下界仍只是理想重叠模型，
不是已经证明可达到的延迟。本轮 insight 是：改变 C++ 的寄存器复用顺序，并不保证 ptxas
保留预期的跨片段交错；必须验证真实依赖顺序，不能仅凭源码假设等待已经被隐藏。

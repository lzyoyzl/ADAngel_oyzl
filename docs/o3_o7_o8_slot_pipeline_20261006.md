# O3/O7/O8：SM80 槽位握手候选（v95）

## 目的与边界

当前最佳仍为 O3 v89、O7/O8 v78 GEMM + v73 preparation。v95 是独立候选，
不修改正式默认、5090、量化/定点语义、G128 scale、全 K 整数安全 guard 或 FP32 回退。
当前有效吞吐模型的必要 MMA 容量下界约为 0.220347 ms（1410 MHz），不是可达性能保证。

此前第五个 producer warp 会减少常驻计算 warp，分散 copy 指令也没有稳定收益。
本次保持 CTA `64×128×128`、128 线程、八条 partial 链；warp0 搬运且计算，四个 warp 都计算。
O3 保持 grouped-M8 和三阶段，O7/O8 保持两阶段；共享内存仅增加128B。

## 数据流与同步

每个槽位两个 SM80 mbarrier：full 等32个 producer lane 的异步 copy 完成；
empty 等128个 consumer thread 完成读取。warp0 初始填充所有槽位。
consumer 等 full 后执行原有 LDSM/MMA/factor 累加，再 arrive empty；
warp0 确认所有读者退出后才覆盖该槽位。其他 warp 可继续读取下一已就绪槽位。

phase 为 `(group/stages)&1`。初始化仅一次 CTA barrier；整数主循环不再每 G128 全 CTA 汇合。
payload 和 factor 的全局字节数、shared layout、16次 LDSM/G128、64次原生 INT4 MMA/G128不变。
集中搬运不等于减少搬运指令：warp0 每组承担更多 copy，可能成为关键路径，必须实测。

同步依据 [CUDA12.8 PTX ISA](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive)：
使用 SM80 支持的 `cp.async.mbarrier.arrive.noinc` 与 `mbarrier.test_wait.parity`，
不用 SM90 `try_wait`、bulk-copy 或 TMA。full 的预期计数包含32次 noinc 异步 arrival；
empty 的 release/acquire 防止读者尚未完成时重写数据。

## 编译与验收结果：不采纳

本地实现已先推送 GitHub，再由 A100 fetch/fast-forward merge。
本地/A100 84项相关单元测试通过；同entry 原生 U4/S4 两路 INT4、cg copy及
旧对照编码一致性审计通过。以下是静态机器码/资源结果，不是 GPU 性能测量：

|指标|O3 v89 → v95|O7/O8 v78 → v95|
|---|---:|---:|
|整数循环静态指令|323 → 436|383 → 526|
|整数循环 local load 条数|0 → 15|0 → 30|
|整数循环全 CTA barrier 条数|1 → 0|1 → 0|
|每 G128 MMA / LDSM|64 / 16 → 不变|64 / 16 → 不变|
|整个 entry 寄存器/线程|168 → 168|168 → 168|
|候选 CUDA 资源查询：CTA/SM|3|3|
|候选 local frame|72B|120B|

**预设编译 gate 均失败。** 去掉全 CTA barrier 的同时，集中 copy 和握手增加了
活跃状态与热循环 spill，未获得更多驻留 CTA。表中静态指令包含分支/等待区域，
不能将它们直接换算成动态执行次数或预测退化百分比；slot 等待也不是“无同步”。

停止在编译/资源筛选阶段，**未启动 v95 目标 kernel**；没有新 GPU Event 延迟、
MSE、NCU 或 sanitizer结果。资源查询只建立 CUDA context 并加载/查询/卸载 CUfunction。
这不是“测试后变慢”，而是“没有通过值得投入完整测试的门槛”。
未修改正式原生扩展，其 SHA-256 仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
保留门槛保护的24样本/四模式驱动，不进行小样本性能筛选，不锁频、不等待 GPU 空闲。

## 当前最佳结果保持不变

|后端|最佳独立候选|既有24样本 GEMM median ms|MSE主参考|MSE median|MSE mean|
|---|---|---:|---|---:|---:|
|O3|v89 grouped CTA / full-K integer|0.437760|O0|0.006653010287410|0.007578847013303|
|O7|v78八链 + v73准备|0.476160|O5|0.005536172273439|0.005053635851003|
|O8|v78八链 + v73准备|0.481280|O6|0.004411084910986|0.004381379299074|

这是此前各自运行的已确认结果，不是 v95 新测量，也不将跨 run 差值当精确 scale 成本。
目标仍未达到，正式默认及5090不变。后续不能直接重复这套集中搬运方案：
必须先证明能控制生产/同步状态的寄存器开销，再投入真实数据性能验证。

[完整编译证据与引用边界](evidence/a100_o378_roof_v95/README.md)。

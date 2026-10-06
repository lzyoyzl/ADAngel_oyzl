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

## 验收状态

待本地契约/协议测试 → GitHub推送 → A100 fetch/merge → 编译/同entry审计。
编译先确认不降低三 CTA 常驻能力、没有明显新增热循环 spill、MMA/LDSM数量不变、
整数热循环没有全 CTA barrier。通过才进行 GPU 正确性、sanitizer 和24样本配对测试。
不进行小样本性能筛选，不锁频，不等待 GPU 空闲，保留所有原始计时和 CV 离群值。
目前没有 v95 性能/MSE结果，不将源码变化算成已实现的加速。

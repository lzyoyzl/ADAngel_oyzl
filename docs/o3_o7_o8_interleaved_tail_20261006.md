# O3 / O7 / O8：N64 边界的软件流水线候选（v96）

状态：仅新增独立候选，等待 A100 编译、审计和实际验收；不切换正式默认或 RTX 5090。

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

## 验收口径

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

# 仅在编译 gate 通过后运行；驱动也会拒绝负 gate。
python scripts/benchmark_o3_interleaved_tail.py --output runs/o378_roof_v96_o3 \
  --warmup 1000 --repeats 200 --rounds 3 --inner 100
python scripts/benchmark_o78_interleaved_tail.py --cubins reports/o378_roof_v96_o78_codegen \
  --output runs/o378_roof_v96_o78 --warmup 1000 --repeats 200 --rounds 3 --inner 100
```

代码先在本地实现并推送 GitHub，再同步到 A100；不修改 `/home/zlouyang` 以外内容。
本文件当前没有新的实测延迟、MSE 或提升百分比。

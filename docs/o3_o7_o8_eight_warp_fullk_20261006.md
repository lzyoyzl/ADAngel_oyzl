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
当前尚无该候选性能结果；实际编译证据完成后追加。

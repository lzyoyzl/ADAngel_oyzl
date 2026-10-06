# O3 / O7 / O8：固定两条 startup 链候选（v97）

状态：独立候选，等待 A100 编译和审计，不修改正式默认或 RTX 5090。

v96 逐条复用旧 partial 槽位，未在 SASS 实现预期交错。v97 在旧八条链之外仅增加两个
high0 startup 结果（8 个额外逻辑寄存器槽位），使下一片段的启动不必先等旧 partial
四个值全部缩放完成；消费旧值后再转移结果，后续计算继续使用原来的32个槽位。
这不是 v92 的完整16链实现，也不做链数参数扫描。

CTA64×128×128、128线程、64个全K accumulator、G128量化/scale/guard、三/两阶段
cp.async.cg、原生 U4/S4 + S4/S4 和 FP32 输出均不变。不增加 MMA、LDSM 或输入复制。
O3保留分组M8，O7/O8仍按原顺序。仅在真实 SASS链峰值9或10、每组64MMA/16LDSM、
分配GPR≤168、hot local指令≤2且旧编码对照通过时，才进入完整24样本测试。
静态链数不等于硬件同时执行数；本轮计数器按SASS出现顺序编号，不把第9链直接当成逻辑N坐标。

通过后直接24样本×三轮，warmup1000/repeats200/inner100；无小样本性能初筛。
GPU资源必须仍支持3CTA/SM；逐位输出、有限FP32、相对O0/O5/O6的MSE、guard/边界等
按原协议验证。仅确认GEMM收益后扩展四模式/有限sanitizer/必要NCU；保留CV失败和离群值。

```bash
python scripts/probe_tail_lookahead_codegen.py --kind o3 --output reports/o378_roof_v97_o3_codegen
python scripts/probe_tail_lookahead_codegen.py --kind o78 --output reports/o378_roof_v97_o78_codegen

# 仅在相应编译 gate 通过后执行；驱动自行拒绝负 gate。
python scripts/benchmark_o3_tail_lookahead.py \
  --output runs/o378_roof_v97_o3_trace24 \
  --warmup 1000 --repeats 200 --inner 100 --rounds 3 --modes compute_only
python scripts/benchmark_o78_tail_lookahead.py \
  --cubins reports/o378_roof_v97_o78_codegen \
  --output runs/o378_roof_v97_o78_trace24 \
  --warmup 1000 --repeats 200 --inner 100 --rounds 3
```

代码须先本地实现和GitHub推送，再A100 fetch/ff-only merge。本文件尚无新的GPU性能或MSE成绩。

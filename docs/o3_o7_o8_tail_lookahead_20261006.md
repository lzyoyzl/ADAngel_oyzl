# O3 / O7 / O8：固定两条 startup 链候选（v97）

## 结论

**A100 编译与原生 INT4 审计通过，但预设排序 gate 失败；不采纳，当前最佳不变。**
本轮没有候选 kernel launch、Event 性能、输出/MSE、NCU、sanitizer或 CUDA 占用查询。
不是实测负加速，也不能把循环指令减少直接报告成速度提升。

## 1. 固定方案与验收口径

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

## 2. 实际机器码结果

|整数主循环静态指标|O3 v89 → v97|O7/O8 v78 → v97|
|---|---:|---:|
|第9条已启动链的首条 MMA 序号|33 → 33|33 → 33|
|已启动未完成链峰值|8 → 8|8 → 8|
|每 G128 原生 S4/S4 + U4/S4 MMA|32+32 → 不变|32+32 → 不变|
|LDSM 指令|16 → 16|16 → 16|
|循环静态指令|323 → 323|383 → 377|
|循环活跃 GPR 峰值|166 → 166|166 → 164|
|entry 分配 GPR / 线程|168 → 168|168 → 168|
|循环 local load/store|0 → 0|0 → 0|

预设的9或10链条件未成立，两套驱动都会拒绝负 gate。O7/O8减少6条静态指令，
但并未落实本轮要验证的跨片段提前启动，不为这项小幅静态变化追加整轮测量。
以上计数来自 `nvdisasm` 存活报告及精确 entry 的 SASS 数据流追踪，不代表动态同时执行数。

先前最佳 entry 的完整编码对照通过。候选 entry 与旧 entry 的完整编码则不同，
即使 O3 的循环计数一致，也不宣称完整机器码完全相同。精确编码比较只归一化 symbol 文本，
不改写 opcode、寄存器、立即数或控制位；不能代替 GPU 正确性验证。
本地和 A100 各124项相关 CPU/source 测试通过；它们不是GPU数值/性能验收。

## 3. 当前已验证最佳（不是本轮新测）

|后端|最佳独立候选|24样本 GEMM median ms|主参考|MSE median|MSE mean|
|---|---|---:|---|---:|---:|
|O3|v89 grouped M8 + 全K整数|0.437760|O0|0.006653010287410|0.007578847013303|
|O7|v78八链 + v73在线准备|0.476160|O5|0.005536172273439|0.005053635851003|
|O8|v78八链 + v73在线准备|0.481280|O6|0.004411084910986|0.004381379299074|

来自先前完整24样本、三轮配对复测；不同run差值不是精确scale成本分解。
本轮没有新conversion、Cold或steady成绩，没有新的候选MSE。正式默认与5090均未改。
当前仍未接近约0.220347ms的乐观必要MMA容量下界；它不是保证可达的时延。

## 4. 复现与证据

源实现先提交并成功推送 GitHub：`8eeeb04e6e5e76ef3a000c10f826501e5bdaef4b`，
再经校验的增量 bundle 在A100 fetch + fast-forward merge。
CUDA12.8.93、SM80、固定 CUTLASS `db1c288993354c88e551c40c19a8fb93a774a241`。
所有本轮服务器写入仅在 `/home/zlouyang/ADAngel_oyzl`。

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

当前编译gate为false，上述GPU测量命令会明确拒绝执行；不要绕过门槛后把未验收候选作为正式结果。

[原始编译、SASS、存活与编码证据](evidence/a100_o378_roof_v97/README.md)。
完整原始归档（含cubin）`tmp/o378_v97_compile_complete.tar.gz` 两端 SHA256：
`90dc034a457955cf486e3654ab201ab36b1a1f0628e8df39483fc26db1a3ebf4`。
正式 `_sm80.so` 仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

本轮证据说明：增加源码结果槽位仍没有增加实际编程交错宽度，下一轮不应继续扫描这类链数写法。

# O3 / O7 / O8：输出 streaming-store 独立候选

## 当前状态

v99 尚未验收，不切换正式默认。最佳仍为 O3 v89、O7/O8 v78 GEMM + v73 准备。
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

实现、编译证据与测试结果将在本节后补齐。

# v40：同 CTA 改用 8 warp，负结果

**结论：不采用。GEMM 最佳仍为 O3/54、O7/O8/59；转换最佳、正式默认和 RTX5090 不变。**

## 改动与资源

固定 CTA `64×128×128`，把 warp 布局从 `2×2` 改为 `2×4`。
不改 G128 scale、两路原生 INT4、FP32 运算顺序或输入格式。
O3 保留三阶段、O7/O8 保留两阶段 `cp.async.cg`，shared-memory 分配不变。
仍在寄存器中重组 `low + 16*high`，B fragment 仍跨 M 复用。
这是独立 cubin，不接入正式调度；控制的完整编码 SASS 与当前最佳 54/59 相同。

|资源|4-warp 控制|8-warp 候选|
|---|---:|---:|
|线程 / CTA|128|256|
|最终 FP32 accumulator / 线程|64|32|
|O3 寄存器 / 线程|168|120|
|O7/O8 寄存器 / 线程|168|128|
|O3 stack / spill stores / spill loads|16 / 12 / 12 B|0 / 0 / 0 B|
|O7/O8 stack / spill stores / spill loads|8 / 8 / 8 B|0 / 0 / 0 B|
|Driver 查询最大活跃 CTA / SM|3|2|
|Driver 查询最大活跃 warp / SM|12|16|

动态 shared-memory：O3 50688 B、O7/O8 34304 B，两种 geometry 相同。
两组 entry 均通过原生 U4×S4、S4×S4 IMMA 与异步搬运审计，没有 INT8 替代。
私有 device body 仅放宽 warp 几何约束；CuTe 根据新布局重新分配 fragment。
因此这不是只改变 occupancy 的单因素实验：供数次数与指令调度也会变化。

## 24 样本结果

24 样本×3轮，warmup50/repeats200；同一样本、同轮循环换序，
统一原生 Driver CUDA Event 计时。432 条 compute-only 记录全部保留。

|后端|控制 median ms|候选 median ms|配对吞吐变化|speedup 描述性95%区间|
|---|---:|---:|---:|---|
|O3|0.476160|0.499456|**−4.80%**|[0.949744, 0.953454]|
|O7|0.500736|0.562176|**−10.73%**|[0.890710, 0.895795]|
|O8|0.503040|0.567296|**−10.77%**|[0.890288, 0.894881]|

配对速度比先在样本内汇总轮次，再跨样本汇总，不直接取表中两个 median 的比值。
共享、未锁频 GPU；CV≥3% 的控制/候选记录分别为 O3 33/36、O7 36/31、O8 36/25，
每项分母72，未过滤。上述区间是当前 trace 的描述性比较，不代表通过严格全阶段稳定性验收。
明显负结果不晋级 cold/steady/conversion 复测，也不声明端到端收益。

|后端 / FP16参考|Median 输出MSE|Mean 输出MSE|相对当前最佳|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位相同|
|O7 / O5|0.005536172426666|0.005053635833762|逐位相同|
|O8 / O6|0.004411084948645|0.004381379215302|逐位相同|

所有被检查输出均为 finite FP32。保留原始 trace/prepared 文件 SHA、量化重放及源格式 provenance。

## NCU：为什么提高驻留 warp 仍然变慢

每个版本另采集一次首个真实样本的 `--set full --cache-control all --clock-control none`。
仅采集 O3、O7；**没有将 O7 的 profiling 当作 O8 的实测**。
通过同一 kernel symbol 及 `launch__block_size=128/256` 核对捕获对象。
NCU duration 不作为上表 Event 延迟，也不与历史 run 的绝对延迟相除。

|指标|O3 控制→候选|O7 控制→候选|
|---|---:|---:|
|NCU duration ms|0.399008→0.418080|0.420576→0.477568|
|动态 warp 指令|99.32M→112.74M|116.29M→130.48M|
|LDSM 指令|4.19M→6.29M|4.19M→6.29M|
|Shared wavefronts|29.62M→38.01M|31.29M→54.65M|
|Local theoretical sectors|3.18M→0|2.10M→0|
|Achieved occupancy|18.00%→24.29%|17.99%→24.32%|
|Eligible warps / scheduler|0.626→0.767|0.775→0.881|
|Issue active|42.46%→45.52%|47.90%→46.14%|
|MIO stall / issue|0.466→0.991|0.545→1.362|
|Long-scoreboard stall / issue|0.764→0.620|0.166→0.177|
|1410MHz 理想重叠容量下界 ms|0.220347→0.234479|0.220347→0.244760|

IMMA、I2F、FFMA 均仍为每项 16,777,216 条；O3/O7 的 FMUL 分别仍为 524,288/16,777,216 条。
**数学工作没有减少，LDSM 反而增加50%。** 更多 N 方向 warp 分担输出，但增加了 A fragment 重复供数。
O7 的 scale 读取/异步搬运代码生成也变化，不能只用 LDSM 增量解释全部差距。
其 source 表中 excessive shared wavefronts 从0.20M增至12.55M，需要按具体指令进一步拆分；
这不是 HBM 字节数，也不能仅凭该总量给所有访问统一贴上 bank-conflict 标签。

PC not-issued 采样中 barrier 占比 O3 8.35%→16.54%、O7 9.96%→18.27%；
MIO 占比9.24%→15.67%、14.02%→25.26%。这些是采样占比，不是运行时间占比；
采样 PC 表示等待的 consumer，不一定是造成等待的 producer。

本轮说明：消除 spill、增加可发射 warp，并不自动提高每单位时间的有效输出。
片段重复读取、额外指令和同步会抵消收益。固定观测工作量的乐观容量下界甚至变差，
新候选主导项转为 L1TEX 数据服务容量；该模型假设理想重叠，**不是承诺可达到的 kernel 时间**。
后续保留当前4-warp和B片段复用，优先减少实际供数/发射工作，避免单独追求更高 occupancy。

## 验证与复现

- 120项 GPU检查：3后端×5种shape×4种pattern×2种geometry；
  K=128/256/384/640/4096，随机、全零、极值、零scale、非默认stream，
  与当前最佳逐位一致，并满足 FP64 语义参考 `rtol=1e-3, atol=1e-3`。
- memcheck、synccheck 各检查一个真实4096³样本×3后端×2geometry，各6条记录、0 errors；
  这是有限范围安全检查，不宣称覆盖所有shape或并发情形。
- 归档前 A100 相关CPU测试260项通过；另有4项归档重算测试覆盖配对/MSE/审计/NCU。
- 正式扩展未重建，SHA-256保持
  `fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

编译/主测源码提交：`72c5d63dfef09915e2bd9cf3c8725fec453c54d5`。
NCU及安全检查提交：`971e278f53a1967c8d03962f31218c326ef7b77c`（CUDA源码相同）。
全部在本地实现、推送GitHub后，再于A100项目fetch/fast-forward。

```bash
python scripts/probe_roof_warp_codegen.py --output reports/warp_recheck
python scripts/audit_roof_warp_probe.py --directory reports/warp_recheck \
  --best-sass docs/evidence/a100_o378_roof_v40/reports/o378_roof_v40/best_controls.sass \
  > reports/warp_recheck/audit.json
python scripts/validate_roof_warp_probe.py --cubins reports/warp_recheck \
  --output runs/warp_validation
python scripts/benchmark_roof_warp_probe.py --cubins reports/warp_recheck \
  --output runs/warp_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/profile_roof_warp_probe.py --directory reports/warp_recheck/ncu \
  --cubins reports/warp_recheck --runs-prefix runs/warp_ncu
python -m unittest discover -s tests/unit -p 'test_roof_v40_evidence.py' -v
```

仓库归档文本、SASS、原始 Event 与NCU导出；cubin、Driver .so、PTX和完整 `.ncu-rep`
保留在 A100 项目的对应 `reports/o378_roof_v40/` 目录，不提交二进制。
传输归档 SHA-256：`93104f90bda1e93b602c30750f014bb9baaa3c2461e4e7d5bd4a72160b28b0d6`。

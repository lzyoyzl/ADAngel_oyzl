# v86：LDSM 与寄存器布局候选的 NCU 对照

**本轮是对 v85 负结果的诊断，不是新的性能优化或默认切换。**
不修改 CUDA、不重新编译，采集 v85 修正版 cubin 中的 v78 控制与 v85 候选。
之前四样本配对 Event 结果仍为 O7/O8 吞吐 −2.65%/−2.91%；不以本轮 NCU Duration 替换它。

## 方法与范围

A100 108 SM，真实 `layer_00_q_proj`，O7、4096³；每个目标 kernel 预热50次，采集1次。
NCU 2025.1.1：`--set full --cache-control all --clock-control none`。
两个报告分别50/49个 replay passes；未锁频的警告保留。不是同一时刻的两条时间线。
本轮不外推到 O3/O8、其他24样本或端到端计时。

原始FP16样本/manifest、prepared一致性、源格式与metadata检查通过。
两入口都恰好2048个CTA、128线程，全部走整数路径；finite FP32、输出对v67逐位相同。
本样本 O7/O5 MSE 均为 **0.00017169781117249843**，对v67输出差MSE为0。
同一实际 kernel 的静态opcode指纹、原生两路INT4、动态指令总量及数学工作量交叉核对通过。
沿用v85的数值和有限sanitizer验收；本轮未扩大安全性覆盖。

## 对照结果

M为百万条动态warp指令或百万个wavefront；不是字节数。

|指标|v78：LDSM|v85：预排布局 / LDS.128|
|---|---:|---:|
|NCU Duration ms（仅诊断）|0.390176|0.401824|
|动态warp指令 M|105.521152|104.398848|
|原生INT4 IMMA M|16.777216|16.777216|
|LDSM M|4.194304|0|
|LDS M（含metadata）|3.907584|8.101888|
|LDS + LDSM M|8.101888|8.101888|
|shared读 wavefronts M（LDS + LDSM）|22.020096|22.020096|
|source shared全部 wavefronts M|38.141952|37.883240|
|LDS/LDSM excessive wavefronts|0|0|
|寄存器/线程；最大CTA/SM|168；3|168；3|
|Eligible warps / scheduler|0.746954|0.713076|
|Issue active %|46.993015|44.346872|
|动态 LDL / STL|0 / 0|0 / 0|

1. **消除 LDSM 不等于减少供数。** 4.194304M条LDSM换成相同数量LDS；
   两类shared读取的wavefront总量完全相同。总动态指令只减少约1.06%，并未提高寄存器允许的驻留数。
   这不是“消除了原有bank conflict”：两者shared读取的excessive计数本来都是0。
2. **没有改善有效发射。** Eligible和issue active反而下降，与v85的Event负结果方向一致。
   当前证据不支持“LDSM本身就是主要瓶颈，所以换成LDS一定快”。
   候选静态入口含fallback spill，但这次动态LDL/STL均为0，不能用spill解释这次变慢。
3. **供数依赖仍需结合调度看。** Not-issued PC样本中short-scoreboard占比
   5.57%→8.59%，wait约36%，math约29%；变化没有表现为等待整体消失。
   这些是不同capture中采样的占比，不是执行时间比例，不证明某条LDS或寄存器bank是因果根源。
   NCU将等待归在消费者PC上，不能把IMMA上的样本全解释为Tensor Core算术耗时。
   [NVIDIA指标解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons-not-issued)

## 对有效吞吐上界及下一步的影响

@1410MHz必要MMA容量下界仍为 **0.220347ms**，对应有效吞吐容量上限约 **623.739 TOPS**。
它是理想重叠的必要约束，不是对当前指令依赖图可达时间的承诺。

|必要资源服务下界 ms|v78|v85|
|---|---:|---:|
|MMA|0.220347|0.220347|
|全部指令发射|0.173235|0.171393|
|L1TEX data wavefront容量|0.182613|0.177340|
|FMA pipe计数器容量|0.141886|0.143527|

上述资源可重叠，不能相加，也不能拿降低某一项推定同等实际提速。
更低的L1容量需求并未转化成更短延迟，下一步不继续扫描LDSM/LDS布局替换。
优先寻找能缩短 **MMA结果→整数加权→跨组累加** 的真实消费链、同时不增加常驻fragment/额外读写的差异；
先用实际SASS确认它与既有候选不同，再决定是否值得GPU测试。尚未发现可承诺收益的新实现。
旧N64/M32候选已经因额外LDSM而失败，单纯缩tile提occupancy也不重复。

## 复核与保留内容

- 采集实现commit：`0be91b30b28078631092ac0843d00a877f079f71`。
- `reports/o378_roof_v86_ncu/`保留原始raw/source CSV、receipt、全部分析、命令和日志。
- receipt内固定v85 cubin/源码/准备库身份；控制机器码与v78一致，见v85证据。
- 正式扩展SHA256仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
- `.ncu-rep`留在A100原路径和本地完整归档，不提交Git。
- 完整归档 `tmp/o378_v86_ncu_complete.tar.gz`，传输前后SHA256：
  `38f688d15dda9d1c81b500282faafb8e1ad09fea5d6a219561dcff78f6f6bfec`。
- 未产生新的conversion/Cold/steady、24样本MSE或Event加速结果。最佳仍O3 v79、O7/O8 v78+v73。
- 本地v86/profile及v85/v80证据回归共19项通过；从原始CSV重新计算分析与指纹，不只检查保存的PASS标签。

```bash
python -m pytest tests/unit/test_register_layout_profile.py tests/unit/test_roof_v86_evidence.py -q
# 已有报告仅重新分析，不启动GPU，不改变capture：
python scripts/run_register_layout_ncu.py --output reports/o378_roof_v86_ncu --analyze-only
```

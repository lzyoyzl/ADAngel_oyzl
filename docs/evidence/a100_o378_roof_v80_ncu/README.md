# v80：当前最佳 O3/O7/O8 的 NCU 与 MMA 链诊断

本轮不改 CUDA kernel，不生成新的 Event 加速比。检查 O3 v79、O7/O8 v78，
准备路径仍分别为 conversion2/GPU guard 与 v73。正式默认、5090不变。

## 方法与身份验证

A100 108 SM，真实 `layer_00_q_proj`、4096³、50次目标symbol预热后采集1次。
NCU 2025.1.1，`--set full --cache-control all --clock-control none`，每次50 passes。
原始/prepared trace校验、GPU metadata guard、finite FP32和输出逐位对照通过。
三个case均2048个CTA走整数路径，无回退；这是单样本诊断，不代替24样本回归。
精确symbol、原cubin SHA、静态opcode指纹、NCU动态指令合计与数学工作量交叉核对通过。

|指标|O3 v79|O7 v78|O8 v78|
|---|---:|---:|---:|
|NCU Duration ms（非Event正式延迟）|0.380320|0.392800|0.388576|
|动态warp指令 M|86.802432|105.521152|105.521152|
|IMMA / LDSM M|16.777216 / 4.194304|16.777216 / 4.194304|16.777216 / 4.194304|
|Eligible warps / scheduler|0.580001|0.747220|0.746675|
|Issue active %|38.8320|47.0028|46.9798|
|Achieved occupancy %|17.9719|17.9421|17.9454|
|寄存器/线程；最大CTA/SM|168；3|168；3|168；3|
|Shared wavefronts M|29.818880|38.141952|38.141952|
|Shared excessive wavefronts M|0.172032|6.422528|6.422528|
|动态LDL / STL|647168 / 32768|0 / 0|0 / 0|

必要MMA容量下界仍为约 **0.220347ms @1410MHz**。它假设理想重叠，
不是已实现的速度，也不保证当前指令依赖图能够达到。以上NCU延迟不能与其他run的Event数值相除构造收益。

## 对下一步有用的发现

1. **不是缺少独立MMA链。** 对每个INT32 C/D fragment进行静态数据流追踪，
   每G128均64条MMA、16条`high→high→×16→low→low`逻辑链，最多8条已开始未结束。
   ptxas已经交错链并提前部分后续B片段加载。这个“8”是程序顺序，不是硬件同时在途数或延迟模型。
2. **仍有发射等待。** 未发射PC样本中wait约36–39%、math约25–29%、barrier约12%。
   大量wait/math样本出现在IMMA消费指令；不能据此将其认定为某条前驱的因果耗时，
   更不能将这些百分比相加当作可以消除的时间。
3. **O7/O8有一个明确的metadata供数热点。** 6,422,528个shared excessive wavefronts
   全部对应A-factor的两处半warp `LDGSTS`：prologue200704、主循环6221824。
   LDSM和普通LDS的此计数为0。该指标不是DRAM字节，也不能直接叫作bank conflict；
   NVIDIA对该派生指标的解释包括未充分参与的线程所产生的额外wavefront。
   因此选取一个完整warp搬运相同因子的独立候选，而不扫描MMA/tile/cache参数。

[NVIDIA NCU指标定义](https://docs.nvidia.com/nsight-compute/NsightCompute/)。

该热点只构成测试动机，不预告加速：即使减少shared工作，必要MMA、168寄存器和3CTA限制仍在。
候选的实际判断见下一轮v81；本轮最佳结果仍为v79/v78+v73。

## 可复核材料

- `reports/o378_roof_v80_ncu/`：三个raw/source CSV、receipt、完整分析、命令和日志。
- `reports/o378_roof_v80_schedule/`：逐条MMA链、寄存器来源、LDSM位置及输入SHA。
- 报告采集源码`4ded55e`；`0bed41f`仅修复审计器对`NOP`/`NOP;`别名应求和而非覆盖的问题。
  原始采集未重跑或修改，静态总指令与动态工作均重新核对。
- 原始`.ncu-rep`、临时driver/cubin保留在A100与本地完整归档，不作为Git源码依赖。
- 完整归档`tmp/o378_v80_ncu_complete.tar.gz` SHA256：
  `b811e9048f93995d6c2eb3d9a31de5f5a1d6218c200f45359e06bd6665adc4b7`。
- 正式扩展SHA保持`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/run_eight_chain_ncu.py --output reports/v80_recheck
python -m pytest tests/unit/test_eight_chain_schedule.py tests/unit/test_roof_v80_evidence.py -q
```

输出目录须不存在。离线`--analyze-only`不重新采集GPU；所有性能结论仍以正式配对Event测试为准。

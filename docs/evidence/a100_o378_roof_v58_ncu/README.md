# v58：全 K 整数累加为什么只有微小收益？

**本轮只做定向诊断，没有新增或切换 kernel，也没有新的普通 Event 加速比。**
复用 v55b 已验收数值的两份 cubin；目标是判断继续减少 I2F 是否值得，避免重复无效候选。

## NCU 实测

A100、CUDA12.8，真实 `layer_00_q_proj`、4096³；两次独立进程，
`--set full --cache-control all --clock-control none`，各只捕获一次匹配 kernel。
控制是 O3/54 的同机器码 cubin；候选是已获准的 v55b 全K整数累加。

|指标|原 FP32 逐组累加|全K整数累加|
|---|---:|---:|
|NCU Duration ms（诊断值）|0.396480|0.394112|
|动态 warp 指令总数|99,319,808|87,891,968|
|IMMA|16,777,216|16,777,216|
|I2F|16,777,216|524,288|
|FFMA|16,777,216|0|
|IMAD|12,492,800|30,818,304|
|LDSM|4,194,304|4,194,304|
|Shared wavefronts（Source口径）|29,622,272|29,622,272|
|Shared excessive wavefronts|0|0|
|Local理论sectors（不是HBM字节数）|3,178,496|8,454,144|
|Eligible warps / scheduler|0.625687|0.557015|
|Issue active %|42.462589|37.855799|
|寄存器 / 线程；驻留CTA上限|168；3|168；3|

I2F减少96.875%，逐组FFMA消失，但整数重建/对齐仍要执行；IMAD约增至2.47倍，
local理论访问约增至2.66倍。总指令减少11.51%，同时发射率下降，因而没有等比例提速。
这支持“工作转移到整数处理与访存等待”，不证明全部差距都由某一种指令造成。

## 对上界和后续工作的影响

|必要容量下界，统一归一到1410MHz，ms|原版|全K|
|---|---:|---:|
|双INT4 MMA工作|0.220347|0.220347|
|I2F工作|0.220347|0.006886|
|全部指令理想发射|0.163055|0.144293|
|Profile条件下的L1数据管线服务|0.177672|0.195690|
|上述完整模型最大值|**0.220347**|**0.220347**|

关键点：**原模型中 MMA 和 I2F 是并列约束，移除大部分 I2F 并没有降低 MMA 必要工作。**
而实际执行仍受依赖、供数、寄存器和调度制约，不能把0.220347ms当作保证可达到的时间。
各容量下界允许重叠，不能相加，也不能用本次NCU时间取代正式Event结果。

PC not-issued采样中，math/long-scoreboard/wait分别由
15.82%/15.48%/32.46%变为19.82%/18.66%/36.58%。这些是等待采样构成，**不是耗时占比**，
不能说消除某类等待就能同比加速。计数与每PC总样本均已对账。

后续筛选据此收紧：

1. 不再把“少一次I2F/FMA”当作充分理由；须同时核对新增IMAD、local访问和发射能力。
2. 不将全K方案直接移植O7/O8：其scale还含整数尾数与双边组变化，不能假设成本更小。
3. 如继续改全K候选，先针对额外整数对齐/临时寄存器供数提出可核验的减量；
   编译后未减少工作或显著增加spill的版本，不投入24样本测试。
4. 不扩大tile/warp/cache扫描，不重启magic，不改INT4或量化语义。

## 正确性、版本和数据边界

本轮被检查的单样本两版输出逐位相同，MSE vs O0均为 `0.0005745973431289735`。
这不是新的24样本MSE验收；全量MSE、36项安全检查及普通Event配对+0.65%仍引用
[v55b原始结果](../a100_o378_roof_v55b/README.md)。本轮未新增conversion/Cold/steady或sanitizer测试。

正式扩展SHA未变：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
两份cubin SHA及NCU调用的manifest、样本、guard、输出检查均保留。
NCU重放期间产生的 `runs/*/results.jsonl` Event值受工具干扰，**不得用作性能比较**。
没有切换当前最佳54/59、转换2/5或正式默认；没有修改RTX5090后端。

复现：现有 `benchmark_roof_fullk_integer_probe.py` 使用 `--samples 1 --rounds 1 --warmup 50 --repeats 1`，
两policy顺序为0、1。NCU只匹配 `adangel_roof_fullk_integer_o3`，控制/候选的
`--launch-skip` 分别为50/101，均 `--launch-count 1`。每次独立输出目录，不覆盖旧结果。
计数器中的I2F/FFMA数量进一步确认没有捕获错版本。

分析器本地实现后push并由A100 fetch/ff-only同步：`ee08dac`。原15项NCU分析回归与2项新测试通过；
归档后另加2项逐值复算/输入范围测试，本地19项全部通过。
[分析JSON](reports/o378_roof_v58_ncu/analysis.json)及raw/source导出可逐项复算；
`.ncu-rep`保留在A100的 `/home/zlouyang/ADAngel_oyzl/reports/o378_roof_v58_ncu/`，不纳入Git。
文本归档SHA256：`ee64c8cc5ab1c115af5cf76cb7fcb7730097d089d76b33620f7af84aa3c07b9a`。

# v52：两组整数对齐的片段复用，仍未超过原最佳

**不采用；GEMM最佳仍为O3/54、O7/O8/59，转换最佳仍为O3/2、O7/O8/4。**
本轮只测试获准的O3相邻两G128，没有扩大窗口，没有修改正式默认或5090。

## 实现与结果

保留v51的准备阶段范围guard、CTA共享对齐参数和精确INT32两组重建。
policy1先计算第一组的两个M片段，保存32个INT32 partial，再复用B寄存器处理第二组。
policy2进一步复用A寄存器，但需在每个N64片段重新读取A。
均为CTA64×128×128、4warp、四个payload槽；不是只修改spill的单因素实验。
policy0的编码SASS与原最佳54完全相同，非安全输入实际派发到通用fallback3。

24样本×3轮×3方案，共216条；预热50、Event测200、单stream循环换序。

|O3方案|Compute-only median ms|配对吞吐变化|speedup 95% CI|CV≥3%记录|
|---|---:|---:|---|---:|
|原最佳54，同轮控制|0.475648|基准|[1,1]|28/72|
|1：复用B片段|0.575488|−17.29%|[0.825471,0.829485]|18/72|
|2：复用A/B片段|0.571392|−16.92%|[0.826168,0.835106]|11/72|

比值来自同样本同轮配对，不直接相除总体median。共享未锁频GPU；离群和CV失败全部保留，
不声称严格稳定性验收通过，不为未观察到的干扰指定具体原因。区间是24样本描述性bootstrap。

三个方案相对O0的输出MSE均为：median **0.006653010287409885**、mean **0.007578847013302749**。
本轮所有输出与原最佳逐位相同；不对其他输入保证舍入结果相同。
真实24样本均通过快速路径guard，最大相邻指数差5，没有以回退冒充候选性能。
每份输入的一次性范围检查wall time单独记录，未计入GEMM Event，不视为免费Cold操作。
GEMM明显退化，故不晋级五轮确认或四模式测试；没有新conversion/Cold/steady结果。

## NCU解释

分别为候选1/2抓取其同轮控制及一个候选launch，`--set full --cache-control all --clock-control none`。
symbol、静态opcode指纹、资源及动态数学工作均核对。下面控制的数学计数在两次抓取中一致。

|指标|原54控制|复用B|复用A/B|
|---|---:|---:|---:|
|NCU duration ms|0.396576 / 0.396672|0.502528|0.500096|
|动态warp指令 M|99.320|118.743|119.669|
|IMMA M|16.777|16.777|16.777|
|I2F / FFMA，各 M|16.777|8.389|8.389|
|IMAD M|12.493|26.157|26.403|
|LDSM M|4.194|4.194|6.291|
|Shared wavefronts M|29.622|31.850|40.239|
|Local理论sectors M|3.178|10.027|3.342|
|Eligible warps / scheduler|0.626591 / 0.626517|0.514275|0.512812|
|Issue active %|42.494 / 42.497|39.570|40.175|
|寄存器 / 线程|168|255|255|
|编译器spill store/load bytes|12/12|72/72|24/24|
|Shared bytes / CTA|50688|68608|68608|
|最大驻留CTA / warp，每SM|3 / 12|2 / 8|2 / 8|

与v51的120/120字节spill相比，源代码复用确实改变了编译结果，但两个候选仍为255寄存器。
复用A/B虽将local工作降至接近原54，却增加50% LDSM，总动态指令比原54多20.49%。
两者仍需更多整数重建、参数读取和更少驻留warp，不能仅凭spill减少预测加速。
Shared excessive wavefront均为0，不是bank conflict证据；sectors不是实测HBM字节。
NCU时间不是正式Event延迟，不混合计算加速比。stall采样也不是耗时分解。

1410MHz、108SM、理想重叠的必要容量下界：

|资源下界 ms|原54控制|复用B|复用A/B|
|---|---:|---:|---:|
|MMA|0.220347|0.220347|0.220347|
|I2F|0.220347|0.110173|0.110173|
|全部指令发射|0.163055|0.194942|0.196462|
|L1TEX数据wavefront容量|约0.17767|0.206323|0.249155|
|模型最大值|0.220347|0.220347|0.249155|

复用A/B甚至使其自己的容量下界变差。后续需要减少浮点转换，同时避免扩大A/B或partial
活跃集合；不能继续通过增加片段重读来换少spill。更大整数窗口仍需用户确认和独立验证。

## 验证与复现

预检、memcheck、synccheck、racecheck各115项，另各8项guard缓存修改检查。
覆盖K128/256/384/640/4096，随机/零/极值/零行scale、指数差13/14/15、code0、奇数尾组。
code255拒绝；危险指数差明确回退；原地修改使guard缓存失效。
code0对FP64参考检查`rtol=1e-5,atol=0`，其他按`rtol=atol=1e-3`。
三个sanitizer均0错误，racecheck同时0警告。最大K4096安全检查形状64×128×4096；
完整4096³由24真实样本验证，不把有限覆盖说成全输入证明。

本地/A100各385项相关CPU测试通过，归档另增5项测试复算性能、MSE、guard、SASS和NCU。
代码commit `5d917cac20b78a0bcc575fac812b036724e4caff`，本地实现push后A100 fetch/ffmerge。
原扩展未重建，SHA保持`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

```bash
python scripts/probe_roof_pair_fragment_reuse_codegen.py --output reports/fragment_recheck
python scripts/audit_roof_pair_fragment_reuse_probe.py --directory reports/fragment_recheck \
  --best-sass docs/evidence/a100_o378_roof_v52/reports/o378_roof_v52/best_controls.sass
python scripts/validate_roof_pair_fragment_reuse_probe.py --cubins reports/fragment_recheck \
  --output runs/fragment_validation
python scripts/benchmark_roof_pair_fragment_reuse_probe.py --cubins reports/fragment_recheck \
  --output runs/fragment_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

输出目录须为新目录。文本归档SHA：`5620481d8096bd1b7cecacce2b78bad27a4c83e9ff8ab6ace5ce1265f519a5d9`。
Event原始序列、SASS、NCU CSV和日志全部保留；cubin与`.ncu-rep`保留在A100项目。

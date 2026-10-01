# v50：O3 两组整数对齐，数值通过，但性能退化

**不采用此候选。原最佳 O3=54、O7/O8=59 和正式默认均不变；RTX5090 未修改。**
本轮经用户确认后，仅测试 O3 相邻两个 G128；没有扩大到 O7/O8 或更大窗口。

## 1. 实现

原始每组整数 partial 与 UE8M0 指数分别为 `P0,e0`、`P1,e1`：

```text
h = min(e0,e1)
Q = P0 * 2^(e0-h) + P1 * 2^(e1-h)
acc = fma(float(Q), 2^h, acc)
```

它保留原 G128 scale，不重新量化、不用一个近似 scale 替代两组 scale。
`|P|≤128×128×8=131072`；指数差≤13 时，合并绝对值界
`131072×(8192+1)=1073872896 < INT32_MAX`。超过范围走原逐组 FP32 路径。
使用有界整数乘法，不对负的有符号整数做 C++ 左移。奇数组尾部单独处理。

保留两路原生 U4×S4/S4×S4、CuTe 坐标、N64 流式片段和最终 A row scale。
CTA 输出仍64×128、4个consumer warp；每组K128，两组配对处理K256。
为保留两组输入并预取下一对，shared 从3个G128槽改为4个槽，
同时保留两组 A/B fragment；这不是只改变求和表达式的单因素实验。

候选独立编译为 cubin，未重建正式扩展。控制版本的编码 SASS 与原54/59完全相同；
O78仅为不变的代码生成哨兵，未进行本轮 O7/O8 性能测试。

## 2. 24样本配对结果

24样本×2方案×3轮，共144条；预热50次、CUDA Event测200次。
同进程同输入交错顺序，所有原始耗时及CV失败记录保留。

|O3方案|Compute-only median ms|配对吞吐变化|speedup 95% CI|CV≥3%记录|
|---|---:|---:|---|---:|
|原最佳54，同轮控制|0.475136|基准|[1,1]|67/72|
|两组整数对齐|1.255424|**−62.23%**|[0.375814,0.379592]|1/72|

配对变化来自同样本同轮比值汇总，不直接相除两列整体median。
共享、未锁频GPU；控制组存在明显波动，不能称为通过CV验收。
但控制72条的median范围0.417792–0.483328ms，候选1.253376–1.262592ms，
两者完全分离，负结果量级远超这轮波动。快照不能确定每个离群值的原因。
bootstrap是当前24样本的描述性区间，不是排除系统噪声的保证。

|O3方案 / O0参考|Median输出MSE|Mean输出MSE|相对原最佳输出MSE|
|---|---:|---:|---:|
|同轮控制|0.006653010287410|0.007578847013303|0|
|两组整数对齐|0.006653010287410|0.007578847013303|0|

本轮24样本所有输出恰好逐位相同；这不代表对任意输入保证逐位相同。
MSE由FP64 reduction重新计算，不是引用历史结果。候选没有性能收益，
不晋级独立确认、转换/Cold/Steady-state测量，更不替换默认。

## 3. NCU：省下浮点指令，却增加更多整数与控制工作

首样本两次单launch，`--set full --cache-control all --clock-control none`；
核对kernel symbol、资源、静态opcode指纹和动态数学工作量。
以下NCU duration不是上表的正式Event延迟。

|指标|原54控制|两组候选|
|---|---:|---:|
|NCU duration ms|0.398976|1.253952|
|动态warp指令 M|99.320|222.839|
|IMMA M|16.777|16.777|
|I2F / FFMA，各 M|16.777|8.389|
|IMAD M|12.493|63.693|
|ISETP M|0.532|25.969|
|SHF M|0.123|18.276|
|BSSY / BSYNC，各 M|0|8.389|
|Shared wavefronts M|29.622|35.914|
|Local理论sectors M|3.178|0|
|Eligible warps / scheduler|0.626735|0.333465|
|Issue active %|42.491|29.764|
|寄存器 / 线程|168|245|
|编译器spill store/load bytes|12/12|0/0|
|Shared bytes / CTA|50688|67584|
|最大驻留CTA / 计算warp，每SM|3 / 12|2 / 8|

LDSM均4,194,304，输入MMA供数总工作未减少；BAR从262,144降为131,072。
shared无excessive wavefront，不应把全部shared流量解释为bank conflict。
wavefront/sector不是实际HBM字节；stall采样不是运行时间占比。

**结论：** I2F/FMA减半确实实现，但逐输出重复的指数比较、对齐、回退分支
及其收敛控制抵消了收益；更多寄存器/共享内存又减少驻留warp。
即使真实trace全部走安全分支，SASS仍执行对应判断与控制工作。
因此“少一半I2F”并不等于“kernel更快”，零spill也不等于更快。

在108SM/1410MHz、理想资源重叠模型下：
I2F必要服务时间从0.220347降到0.110173ms，MMA仍需0.220347ms；
但全部指令发射下界从0.163055增到0.365837ms，成为本候选的最大容量约束。
候选自己的下界比原0.220347ms更差，不是向原目标靠近，且不是可实现延迟保证。

下一步合理候选：将指数差安全检查提到计时外的准备/分派阶段，
将对齐参数每CTA每列每对group生成一次并复用，安全快速路径不保留逐输出回退分支；
不安全输入仍明确回退。准备成本需单独记录，不能从端到端计时中隐藏。
该后续版本尚未实现/测量，本轮不能据此宣称收益。

## 4. 验证、来源与复现

预检、memcheck、synccheck、racecheck各75项：K128/256/384/640/4096、
随机/全零/极值/零row scale、指数差13/14/15、UE8M0 code0，含非默认stream。
原54拒绝code0，故5项code0只测候选，对FP64参考检查，`atol=0,rtol=1e-5`；
其余35组输入×两方案满足`rtol=atol=1e-3`。code255在host拒绝、不提交GPU。
三项sanitizer均0错误，racecheck同时0警告；其K4096形状为64×128×4096，
完整4096³来自24真实样本验证。最初因54拒绝code0而中止的预检日志也保留。

本地/A100各368项相关CPU测试通过；归档另加复算测试。
CUDA源码commit `c311ac36000c349caf950f9f610396de8c439a3d`；
最终测试/NCU脚本 `b130160d1beaf70f759eb5fe831be4826e10a1e9`。
均先本地实现并push，再A100 fetch/ffmerge。正式扩展SHA保持
`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

```bash
python scripts/probe_roof_pair_alignment_codegen.py --output reports/pair_alignment_recheck
python scripts/audit_roof_pair_alignment_probe.py --directory reports/pair_alignment_recheck \
  --best-sass docs/evidence/a100_o378_roof_v50/reports/o378_roof_v50/best_controls.sass \
  > reports/pair_alignment_recheck/audit.json
python scripts/validate_roof_pair_alignment_probe.py --cubins reports/pair_alignment_recheck \
  --output runs/pair_alignment_validation
python scripts/benchmark_roof_pair_alignment_probe.py --cubins reports/pair_alignment_recheck \
  --output runs/pair_alignment_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

原始Event/MSE/环境/SASS/NCU CSV及日志在本目录；二进制保留在A100项目。
文本归档SHA：`3986373c3037037b2b8c8409d653ad86f1ecf6736edfc59f08e0d8f4a7efeece`。

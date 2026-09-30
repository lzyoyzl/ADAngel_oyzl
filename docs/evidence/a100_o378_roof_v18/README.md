# v18：更多驻留 CTA 的寄存器预算消融

源码`404b392`；CUDA12.8，A100-PCIE-40GB。正式默认不变，候选28不采纳。
二进制SHA-256：`d4b50af1e92adae9b758d31a2291976b70fe33ce353cee48e9a9fc5e31010249`。
归档SHA-256：`edec83ea3bb6a6cd760a327736913c920cc4673b8a502f6e132ec249de36f636`。

## 改动与正确性

28与22共用完全相同的device计算体，仅将launch-bound从`(128,3)`改为`(128,4)`。
保留64×128×128 CTA、K128两阶段cp.async、B fragment复用、两路原生INT4、逐G128原顺序FMA。

- 78实例指令审计通过（允许spill），12个正式实例、75个旧候选的编码SASS不变。
- 新3实例均128寄存器；O3线程栈136B，O7/O8为160B，存在LDL/STL，不能标为零spill。
- 522项合成逐位正确性检查通过；memcheck/synccheck各126项、0错误；有限K768 racecheck为0 hazards/errors/warnings。
- 一个真实样本三轮四模式108条均逐位相同，组内MSE不变；这不是24样本验收。

## 平衡顺序合成初筛

4096³，5种实现×5轮，每个实现各执行位置一次；warmup50、repeats200，保留全部CV离群记录。
下表为compute-only median ms，不混用NCU重放时间。

|后端|旧正式|对照6|K128两阶段22|K128三阶段23|四CTA预算28|
|---|---:|---:|---:|---:|---:|
|O3|0.557056|0.533504|0.530432|0.687104|0.744448|
|O7|0.599040|0.560128|0.542720|0.535552|0.800768|
|O8|0.598016|0.560128|0.542720|0.535552|0.808960|

各列CV≥3%的记录数/5：O3 `1/3/2/0/1`，O7 `2/2/1/3/0`，O8 `2/1/2/1/0`。
候选28明显更慢，真实样本四模式烟测也呈相同退化，因此不继续投入24样本正式性能验收。
数值正确不意味着性能候选可采纳。

## O7 NCU：驻留收益被溢出代价抵消

`--set full --cache-control none --clock-control none`，单独profiling，不与Event测量并行。

|指标|候选28|
|---|---:|
|NCU Duration ms|0.713952|
|寄存器/线程；最大CTA/SM|128；4|
|Achieved occupancy|23.855%|
|Eligible warps/scheduler|0.531331|
|Issue active|36.519%|
|动态warp指令|138,256,384|
|Local理论sectors|61,079,552|
|Shared wavefronts|39,190,528|
|Shared excessive wavefronts|8,388,608|
|重算资源服务下界 ms @1410MHz|0.322498|

实际查询证实驻留上限升至4CTA，但溢出读写和总指令增加，发射率并未提升。
模型最紧约束回到L1TEX服务容量；不能继续拿候选22/23约0.220347ms的较低工作量模型套在28上。
该下界假定理想资源重叠，并非保证可达时间；local理论sectors也不是实际HBM流量。
Shared excessive仍在LDGSTS，不能归咎于LDSM。

## 数据位置

`runs/o378_roof_v18_budget_screen`保存完整初筛；`budget_four_smoke`保存单真实样本四模式/MSE；
`memcheck/synccheck/racecheck`保存有限范围安全检查；`ncu_o7_t28`记录profiling上下文。
`reports/o378_roof_v18`有审计、旧代码对照、NCU原始CSV及工作量重算。完整二进制、SASS/PTX和
`.ncu-rep`保留在服务器项目同名目录，不纳入Git。没有更改5090。

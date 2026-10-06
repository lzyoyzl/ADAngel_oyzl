# O3/O7/O8：小 INT4 atom 容量筛查（v102）

## 目的与查重

当前最佳仍为 O3 v89、O7/O8 v78+v73，v99 仅是微小正向候选；正式默认和5090不改。
v101 的 CTA 时间线不支持将调度尾部视为主要突破口。

本轮核对迭代简报及源码后，仅筛查此前未测试的 INT4 `m8n8k32` atom。
它不是此前 O4 的 binary `m8n8k128`，也不是再次扩大 CTA、warp 或 partial 数组。
此前 v92 的16链需要64个 partial寄存器且驻留CTA减少；本轮小atom每链只有2个C寄存器，
在32个逻辑partial寄存器内可表示16条源码链，但同一物理工作量需要4倍MMA指令。
这一代价可能抵消收益，不能以源码链数推断硬件并发或性能。

不重复 v37/v38 缓存、v41/v95 producer/握手、v44/v45/v46地址、v47/v48几何与stream、
v83/v84 tile/stage、v87 coefficient shift、v92/v94/v96/v97/v98 chain/资源、v100数字表示等。
新诊断不改变量化、G128数学或两路INT4要求；不替换正式 kernel。

## 同工作量诊断

公共 CUDA12.8 [PTX fragment定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-8832)
列出整数 `m8n8k32` 的 A/B 各1个32位寄存器、C/D各2个INT32寄存器。
具体是否得到原生INT4 SASS及容量收益，必须在A100实际审计和测量，不能只引用PTX。

|项目|匹配控制|小atom诊断|
|---|---:|---:|
|MMA shape|16×8×64|8×8×32|
|partial源码槽位/线程|32|32|
|源码独立链/片段|8|16|
|MMA/warp/组|64|256|
|CTA、线程、组数|2048、128、256|相同|
|驻留容量|shared padding限定3 CTA/SM|相同|
|物理INT4操作量|2,199,023,255,552|相同|
|每组checksum加法|64，观察全部D寄存器|相同|

各做50次预热、200次单stream CUDA Event，交错两种顺序，不筛离群值、不锁频、不等待空闲。
6个常量nibble模式×2种组数×2种kernel共24项checksum验证，包括负值和边界。
验证仅证明此指令诊断的常量输入数值，不等于真实矩阵的fragment坐标或24样本MSE验收。
SASS检查同entry两路原生S4/S4、U4/S4及精确动态循环工作量，热循环不得有内存读写。

与v82不同，本轮控制观察全部D寄存器，不能把其绝对时间与v82拼接。
除以8的32组等效时间仅用于工作量归一化；没有真实payload、scale、全K accumulator和FP32输出，
不是正式GEMM成绩、可达到的kernel peak或相对当前最佳的加速。

## 预设投入门槛

只有本轮配对容量比的bootstrap95%区间下界超过1.10，才值得实现完整GEMM候选。
未达到就停止，不重测到通过，也不扫描更多atom/chain配置。
此处统计单位是连续诊断launch配对；区间不是24样本普适性能结论。

运行方式（只在固定A100项目目录内生成诊断文件）：

```bash
python scripts/probe_small_atom_capacity.py --output reports/o378_roof_v102_capacity --run
python -m pytest tests/unit/test_small_atom_capacity.py -q
```

## 结果状态

当前为诊断实现与预设门槛，服务器结果尚待记录。没有新的GEMM、MSE、conversion或端到端成绩。

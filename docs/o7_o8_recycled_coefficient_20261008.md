# O7/O8：受限系数窗口与操作数寄存器交接（v141）

状态：独立编译候选，尚无新的性能/MSE结论，正式默认与5090不变。

## 为什么仍值得做一个编译验证

当前v78会将`partial*A_factor*W_factor`安排为两次依赖partial的整数运算。
系数乘积本身与MMA结果无关；若在MMA完成前准备它，最后只需一次加权更新。
这一原则适用于双侧分组scale，不依赖NVFP4/HiF4的码值或24样本的特殊分布。

不是声称“提前算系数”从未测过：v72在原v67分离high/low partial路径上固定系数乘积，
出现4条热local读取，没有确认性能收益；v76/v132的完整系数表也有额外供数成本。
本轮区别是：在当前八链路径上，显式借用**已被全部low-K64_0 MMA消费完的B0片段**，
只保留8个系数（两M atom×一N atom×4个结果），不新增完整系数表或HBM/shared搬运。

`recast`只表达源码中的存储复用，**不保证物理寄存器减少或延迟变短**，必须审计实际机器码。
O3只有一个组因子，不采用这个双侧乘积方案。本轮不扫描窗口大小或改变tile。

## 不变与改变

- 不变：两路原生INT4、G128/全K安全guard及FP32回退、量化、输入布局、转换、
  64×128×128 CTA、4 warp、2stage、8条MMA链、32 partial、64最终accumulator和输出。
- 改变：B0最后一次读取后先准备第一批8个系数，再发出最后一波low MMA；
  此后按N atom逐批消费/覆盖这8个槽位。
- 保留既有系数/乘积/累积INT32范围检查；不允许用溢出或更改scale换取加速。

## 编译前规定的门槛

必须同时满足：同entry原生S4/U4两路各32条MMA、LDSM16/异步copy10不增加、
标量shared读取不增加、分配寄存器≤168、整数热循环无local load/store、静态指令≤v78，
并确认全部64个结果使用独立系数乘积更新，保留8条逻辑MMA链。

这些只判断是否值得执行，不是速度预测。通过后先做正确性/边界/内存安全，
再直接进行24样本×3轮配对、MSE与相关计时验证；不做小规模性能初筛。
不通过就停止，不把旧系数候选换名继续扫描。

本地单元验证及A100独立编译命令：

```bash
python -m pytest tests/unit/test_recycled_coefficient_codegen.py -q
python scripts/probe_o78_recycled_coefficient_codegen.py \
  --output reports/o378_roof_v141_recycled_coefficient_codegen
```

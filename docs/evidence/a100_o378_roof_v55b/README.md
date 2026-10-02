# v55b：全K候选写回修正与24样本结果

**结论：O3全K整数流式累加只有微小GEMM收益，保留独立候选，不改正式默认。**
实现commit `504a2251e05155ed06c3ffaab6804e5429357786`；原版v55为
`6be5dba4839ca02351947999357f7f4f515fa5ee`，其负结果未覆盖。

## 各版实测

|版本 / 覆盖|同轮旧最佳O3/54 ms|全K候选 ms|配对吞吐变化|
|---|---:|---:|---:|
|v55，4样本×3轮|0.451584|0.472576|−4.57%|
|v55b，4样本×3轮|0.451840|0.448512|+1.31%|
|v55b，24样本×3轮|0.474112|0.470784|+0.65%|

均为4096³，warmup50、repeats200、单stream，控制/候选共用原生Driver CUDA Event，
循环交换次序；输入准备/分配/CPU guard不在计时内。表内ms先取各样本跨轮median再取样本median。
加速比先逐样本配对再聚合，因此不要求等于表中两个总体median之比。不能跨run直接相减推导收益。

24样本配对加速比 **1.006522×**，bootstrap median 95% CI **[1.002227,1.010439]**。
旧/新各72条记录中CV≥3%分别 **35 / 37**，均完整保留。未锁频且有明显时间波动，
不能称为严格稳定性验收通过，也不据此断言额外负载是唯一原因。
小样本/全量方向一致，但收益仅约千分之六，不值得继续为此投入端到端集成与窗口扫描。

## MSE与正确性

|24样本输出MSE，相对O0|旧最佳O3/54|v55b|
|---|---:|---:|
|Median|0.006653010287410|0.006653010287410|
|Mean|0.007578847013303|0.007578847013303|

全部72条候选输出与旧最佳逐位一致，彼此MSE为0，max absolute difference为0。
这只说明已测真实样本；该算法一般会改变FP32舍入顺序，不保证任意输入逐位相等。
36项合成语义测试覆盖随机、全零、INT8/INT4极值、行列组scale、零行scale、
大且安全的整数范围、危险指数差回退、K256回退和非默认stream。
同一scale tensor原地修改的2项缓存失效测试通过；code255在launch前拒绝。

memcheck、synccheck、racecheck各重跑36项，均0 errors，racecheck还为0 warnings。
sanitizer只过滤到本轮probe符号，最大检查形状128×256×4096；
完整4096³由24样本数值测试覆盖，不宣称对所有输入的安全证明。

## 数学与实现

对每个权重列选择 `h=min(e[0..31])`：

```text
I = 0                         # INT32，替换旧FP32 accumulator
逐个G128：
    P = INT4(U4低位 × Q4) + 16 × INT4(S4高位 × Q4)
    I += P × 2^(e[g]-h)       # 各组原scale独立保留，整数域精确
Y = float(I) × 2^h × A_scale  # 只在最后转FP32
```

`131072 * sum(2^(e[g]-h)) <= INT32_MAX`保证所有有符号乘积、任意累加前缀都不溢出，
使用Python任意精度整数检查该界。24样本的最大界为81,657,856。
只支持guard通过的K4096/正常UE8M0；范围不安全或其他K回退控制。并未新增量化或抹去组scale。
仍使用64×128×128 CTA、4个warp、3阶段异步搬运；每次只有一组A/B fragment活跃。

v55b只修正最后写回：先完成全部scale读取与浮点转换，再写输出。
用`__float_as_int`/`__int_as_float`复用原32bit寄存器槽，而非新建第二套64元素accumulator；
这些是bitcast，不是浮点到整数的数值转换，不增加量化误差。

## 为什么没有大幅提速

|资源或静态SASS指标|旧最佳控制|v55原版|v55b|
|---|---:|---:|---:|
|寄存器/线程|168|168|168|
|驻留CTA/SM|3|3|3|
|动态shared bytes/CTA|50688|50688|50688|
|ptxas spill stores / loads bytes|12 / 12|36 / 36|36 / 36|
|静态LDG指令数|7|166|26|
|静态IMAD指令数|142|311|223|
|静态LDSM / IMMA指令数|16 / 64|16 / 64|16 / 64|

原版把scale读取与store交错，使编译器难以排除指针别名，出现重复scale加载。
v55b明显减少这部分工作；但逐G128仍须计算并乘整数对齐因子，两个INT4点积和fragment供数不减少，
spill比旧版本更大。**省下I2F/FMA，不等于这些成本全部消失，也可能把压力转移到整数管线。**
这是源码/SASS与资源支持的解释，不是NCU量化的瓶颈占比；静态opcode数不能直接当动态执行次数。

同一候选入口包含原生 `IMMA...U4.S4`、`IMMA...S4.S4` 和 `LDGSTS.BYPASS`，
无INT8 MMA退化。控制与原O3/54、O7/O8/59机器码逐字相同，O7/O8哨兵未变。

## 计时范围与停止条件

CPU范围检查和锚点构造由内部probe缓存，准备wall time逐样本保存为
`guard.preparation_wall_ms`，不藏在GEMM中，也不把它算作免费转换。
24样本该CPU探针准备实测约26.09–87.91ms（均值32.15ms），显然不能直接用作正式Cold路径。
本轮没有conversion、Cold或steady实测，不能宣称端到端提升。
如果将来接入正式路径，必须优化并计入这一准备步骤；当前微小收益不足以支持继续投入。
正式默认、既有完整最佳组合、RTX5090实现均不修改。

## 证据与复现

- `runs/o378_roof_v55b_screen/`：4样本原始Event/MSE、环境、summary。
- `runs/o378_roof_v55b_trace24/`：24样本全部144条记录、28,800个Event样本及GPU快照。
- `reports/o378_roof_v55b/`：PTX/SASS、编译资源、audit、preflight和三类sanitizer。
- 下载归档SHA256：`1201872182dd545bb84c44b6876343e9c338206f0f49abda416735c643f5a975`。
- 原生扩展未重建，SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
- CPU证据测试重新计算全部统计/MSE汇总，并将控制SASS与v54存档逐字核对。

```bash
python scripts/probe_roof_fullk_integer_codegen.py --output reports/o378_roof_v55b
python scripts/audit_roof_fullk_integer_probe.py --directory reports/o378_roof_v55b \
  --best-sass reports/o378_roof_v54/after.sass > reports/o378_roof_v55b/audit.json
python scripts/validate_roof_fullk_integer_probe.py --cubins reports/o378_roof_v55b \
  --output reports/o378_roof_v55b/preflight
python scripts/benchmark_roof_fullk_integer_probe.py --cubins reports/o378_roof_v55b \
  --output runs/o378_roof_v55b_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

目录必须新建。三类安全检查使用`compute-sanitizer --tool <tool>`，
附`--kernel-name kns=adangel_roof_fullk_integer_ --error-exitcode 99`，运行相同validate脚本、各用新输出目录。

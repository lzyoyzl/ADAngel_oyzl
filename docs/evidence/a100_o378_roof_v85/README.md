# v85：预排 MMA 寄存器布局，隔离供数候选

**修正版正确性通过，但四样本三轮GEMM未获益，停止候选。** O7/O8配对吞吐−2.65%/−2.91%，
输出/MSE逐位不变；最佳仍O3 v79、O7/O8 v78+v73，不改正式默认或5090。

## 实现与正确性修复

首版错误发生于O7/random/candidate/conversion-only后的输出检查，尚未进入真实样本计时。
packing本身通过逐nibble oracle，但同一consumer warp的相邻N8 fragment在逻辑N上间隔16，
不是相邻8列。首版将后一fragment读成了另一N warp的列；因此只验证packing可逆不够。
修正W为N32、两个warp分别打包其自身两个N8 fragment；补充使用实际CuTe
`partition_A/B/C`逐坐标验证producer到consumer的主机程序，并检查它能识别旧映射错误。
原始失败的`reports/o378_roof_v85_codegen`和空screen目录保留，不覆盖；修正版使用新目录。
主机CuTe verifier穷举A的16,384、B的32,768个consumer nibble坐标全部正确，
同时识别旧映射24,576处B坐标错配。GPU另按独立oracle比较全部packed nibble和最终输出。

v80的等待样本仍集中在MMA消费点，v83/v84未改善速度。本候选不再扩大tile或stage，
而是把G128 packed payload无损排为每lane连续16B的MMA fragment，整数主循环改用
`ld.shared.v4.b32`，保留原cp.async、barrier、64×128×128、4warp、两阶段、八MMA链、
整数factor/guard和FP32 epilogue。这是供数假设，不意味着LDS必然比LDSM更快。

映射由pinned CUTLASS的CuTe `partition_A/B`构造；另有独立位排列oracle、完整payload与
输出对照，不能只凭公式或CPU槽位模型认定GPU正确。A每个M16×K64、W同一warp拥有的
两个N8×K64各产生32lane×16B，数据量不变；scale布局和数值完全不变。

初版显式增加A/W各一次重排kernel，并保留自然payload以供旧FP32安全回退。
4096³额外预分配24MiB，不在计时内申请；**重排不是免费离线工作**：
conversion-only与Cold计入W和A重排，steady只计A重排，compute-only提前准备。
复用v73原Event计时体，唯一新增执行是对应转换阶段的重排；正式默认不切换。
转换成本已在实现中计入对应计时路径，但本轮只作cached GEMM性能初筛；
不因四模式正确性检查通过，就把其0预热/2次检查时间当作端到端成绩。

## 四样本三轮配对结果

首层q/k/v/o四个真实4096³样本，warmup50、repeats200、单stream、预分配、交错顺序，
48条记录、9,600个原始Event区间全部保留。原始FP16直接量化为源格式，公共准备不计时。
ms是样本内三轮median再取跨样本median；配对提升先逐样本取speedup，不能直接用表中两列相除。

|后端|v78控制 ms|v85修正版 ms|配对吞吐变化|描述性speedup 95% CI|CV≥3%控制/候选（各12）|
|---|---:|---:|---:|---|---|
|O7|0.444928|0.458752|−2.65%|[0.957399,0.977247]|12/12|
|O8|0.440832|0.455936|−2.91%|[0.963048,0.975336]|12/9|

共享、未锁频GPU；CV失败不删除，不认定严格稳定性通过，也不凭快照猜测每个离群原因。
这些是相关的四个首层样本，不代替24样本总体。没有新的conversion/Cold/steady成绩。
不扩大24样本、不新增NCU、不为负结果开发融合重排或迁移到O3。

|输出MSE参考|Median|Mean|相对v67/v78|
|---|---:|---:|---|
|O7/O5|0.000102067943158|0.000099167492985|全部输出逐位相同，差值MSE=0|
|O8/O6|0.000123493261233|0.000122317652713|全部输出逐位相同，差值MSE=0|

48条均finite FP32、metadata精确一致；四个样本的CTA全部走整数路径。
以上MSE只来自四样本，不替换之前24样本的MSE汇总。

## 指令与资源分析

|项目|v78控制|v85修正版|
|---|---:|---:|
|寄存器/线程、活跃CTA/SM|168、3|168、3|
|CTA / shared / stages|64×128×128 / 34304B / 2|相同|
|整数G128循环静态指令|383|379|
|两路原生INT4 MMA / warp / G128|32 S4+32 U4|相同|
|操作数shared→register指令|16 LDSM.16.M88.4|16 LDS.128|
|整数循环local load/store|0/0|0/0|
|编译器报告stack、spill store/load|0、0/0B|8、8/8B|

同一候选entry PTX/SASS保留U4×S4、S4×S4及cp.async/LDGSTS，没有INT8退化；
新cubin内控制入口编码SASS与既有v78逐条一致。v85静态spill位于FP32回退路径
（循环有2条LDL），不能将其解释成这四个全整数样本变慢的主因。
回退源码数学未改，但其机器码重排了，不能称整个fallback SASS逐条不变。

**结论：预排fragment并不自动减少供数工作。** 必要MMA、payload字节数和操作数load数量均未下降，
寄存器分配/驻留也未改善；4条静态指令的减少没有转化为延迟收益。
本轮无新增NCU，不把这些静态相关性包装为因果耗时分解，也不推断LDS对所有布局都劣于LDSM。

## 验证与复现

64项GPU预检覆盖两后端、两路径、四种计时模式以及随机/全零/交替极值/宽scale，
另12项边界检查；包含非默认stream和全CTA回退。均对v67逐位一致，
对FP64语义参考满足rtol/atol=1e-3。规模为64×128×4096及128×256×4096，
不是4096³ sanitizer覆盖；真实4096³输出另由上述48条记录验证。
memcheck、synccheck各0 errors，racecheck 0 errors/0 warnings；各自完成同样64+12项有限检查。
完整原始Event、GPU快照、provenance、指令/资源和三份安全日志随本目录归档。
完整下载归档SHA256：`d040958b5fce65fb9e6401125baebac4eca0e6be8601b23de12350288409f68d`。

实现commit `d950b00`，映射修复commit `258e053`，均本地编辑、GitHub推送后在A100 ff-only合入。
正式扩展SHA256仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/probe_o78_register_layout_codegen.py --output reports/o378_roof_v85_codegen_checked
python scripts/benchmark_o78_register_layout.py --cubins reports/o378_roof_v85_codegen_checked \
  --gpu-build reports/o378_roof_v85_codegen_checked --output runs/o378_roof_v85_screen_checked \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
```

复测须改用尚不存在的输出目录，并给`--cubins`与`--gpu-build`传相同新目录。

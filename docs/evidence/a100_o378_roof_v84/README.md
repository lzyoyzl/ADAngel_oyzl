# v84：全K整数路径的三阶段前瞻，独立候选

**不采纳。** 四样本三轮，O7/O8配对吞吐−3.32%/−3.47%，输出逐位一致、MSE不变。
不扩大24样本/四模式，不继续扫描stage数量；最佳仍O3 v79、O7/O8 v78+v73，正式默认和5090不变。

## 配对结果

A100，首层q/k/v/o真实4096³；从原FP16 trace直接量化源格式，50预热/200次单次CUDA Event，
三轮、单stream、预分配、交错顺序；控制/候选共用原v73准备与原生计时驱动。

|后端|同轮v78 ms|三阶段 ms|配对吞吐变化|描述性speedup 95% CI|CV≥3%，旧/新（各12条）|
|---|---:|---:|---:|---|---:|
|O7|0.445440|0.460800|−3.32%|[0.952703,0.969365]|12 / 12|
|O8|0.440320|0.457728|−3.47%|[0.956916,0.972851]|11 / 11|

48条记录/9600个Event区间全部保留，未锁频共享GPU；没有通过严格CV<3%门槛。
延迟按样本跨轮median后再跨样本median，speedup先逐样本/轮配对，不能直接用表中两列延迟相除替代。
四样本不是24样本最终结果，不能据此替换历史正式成绩。

|仅四样本的输出MSE|旧/新 median|旧/新 mean|新旧差MSE|
|---|---:|---:|---:|
|O7 / O5|0.000102067943158|0.000099167492985|0|
|O8 / O6|0.000123493261233|0.000122317652713|0|

所有真实输出finite FP32、与v67/v78逐位一致，metadata检查与MSE回归通过。
这次只测cached compute-only，没有新的conversion、Cold、steady性能结果。

## 测试依据与唯一变化

v80显示O7/O8仍有供数/发射等待；v83扩大tile虽减少加载，却增加寄存器并降低驻留CTA，未获益。
这里保留v78的64×128×128、四warp、八条MMA链和全K整数guard，仅将整数路径从两槽改为三槽，
以相同每组加载工作量更早发出下一组数据。不重启旧FP32路径的stage/tile扫描。
原先每轮只有下一组前瞻，本候选先提交group0和1；处理group g时允许g+1仍在途，预取g+2。

`cp.async.wait_group 1`等待除最新一组以外的拷贝完成；末组用`wait_group 0`排空。
随后保留CTA barrier，保证跨线程可见性，并确认group g−1的读者结束后才能复用其槽位。
不能简单删除wait或barrier。依据：[NVIDIA PTX等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)。

shared从34304增至51456B，launch bound仍128线程/3CTA；Driver确认实际驻留上限仍3CTA。
不新增payload/scale量化、系数操作或输出；unsafe CTA保留原两阶段逐组FP32回退。
转换与范围证明仍为v73，不能将新增guard准备排除在端到端时间外。

## 实际指令与资源

|项目|v78两阶段|v84三阶段|
|---|---:|---:|
|寄存器/线程；最大CTA/SM|168；3|168；3|
|Dynamic shared B|34304|51456|
|Stack / spill store / spill load B|0 / 0 / 0|8 / 4 / 4|
|整数循环静态指令|383|399|
|循环MMA / LDSM / LDGSTS条数|64 / 16 / 10|64 / 16 / 10|
|整数循环local load条数|0|1|

包含的v78控制机器码完全一致；候选同entry两路原生INT4、cp.async与wait1/0审计通过。
三阶段没有减少必要MMA或payload工作，却新增了槽位寻址、分支和一次热循环local读取。
这是解释负结果的代码证据，**本轮没有NCU因果耗时分解，不能说每一项分别导致多少延迟**。
原两阶段FP32回退的数学源码没有改，但在新entry内的编译调度发生变化：循环440→453条；
不能声称候选内部回退机器码不变。数值预检覆盖该路径。

## 正确性、安全范围与复现

64项GPU检查覆盖O7/O8、随机/零/正负极值/宽scale、两配置、四种模式，
形状64×128×4096和128×256×4096、非默认stream；逐位参考与FP64语义容差均通过。
另有12项非法编码/边界guard检查。CPU槽位模型只检查调度，不代替GPU同步验收。
memcheck、synccheck、racecheck各自完成同样的64+12项检查，最终日志均0 errors，
racecheck也0 warnings；宽scale回退路径逐位检查通过。已核对完整终态日志，
这不是4096³ sanitizer或任意shape验收。

```bash
python scripts/probe_o78_three_stage_codegen.py --output reports/o378_roof_v84_codegen
python scripts/benchmark_o78_three_stage.py --cubins reports/o378_roof_v84_codegen \
  --output runs/o378_roof_v84_screen --samples 4 --rounds 3 --warmup 50 --repeats 200
```

需要现有v67/v73/v78构建和原FP16 trace；输出目录必须全新。
CPU槽位模型覆盖更多尾长，但实际GPU候选仍只支持K4096；不能据此声称其他K的运行验收。
实现/编译/测试提交为`f709a77`；先本地实现并push GitHub，再由A100 bundle fetch/ff-only merge。

完整传输归档SHA256（服务器与本地一致）：
`fc0f308f87b95fe2418f5569cdbf8d5c3be0bf1723695eba5b598f25eded2984`。
正式`_sm80.so`仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，
没有重编译或切换默认。Git保存原始Event、provenance、PTX/SASS、liveness和安全日志；
完整归档的cubin仅留本地及A100，不作为Git源码依赖。

四项离线证据测试重新计算48条计时与配对区间、MSE、源码/产物SHA、同entry指令、
新旧循环静态工作和安全覆盖范围；不把编译/模型检查当作性能或完整GPU验收。

```bash
python -m pytest tests/unit/test_o78_three_stage.py tests/unit/test_roof_v84_evidence.py -q
```

# v77：单位 scale 成本隔离，不是新的 O7/O8 性能版本

## 目的与严格边界

保留首层四个真实样本的 O7/O8 **整数 payload**，但将所有 scale、factor、base 改为1。
在相同的修改后输入上，对照通用 v67 和编译期 `coefficient=1` 专用入口。
它回答“省去逐组系数算术后，这套组织方式还需多少时间”，**不保留原实验的数值语义**，
不能把以下时间、比值或零MSE写入原O7/O8结果，也不作为可上线候选。

两路 U4×S4 / S4×S4 MMA、CTA64×128×128、4warp、两级cp.async、payload搬运、
factor搬运、最终累加和写回保持源码结构。实际编译调度与spill仍会变化，故这不是严格的
scale时间分解、Amdahl比例或所有等价kernel的性能上界。原约0.220347ms必要MMA容量下界不变。

Host拒绝非单位scale/factor/base或非零status；检查在Event外。
K4096下整数点积及任意前缀的幅度不超过`4096×128×8=2^22`，INT32与FP32最终表示均精确。
reference用FP64整数矩阵乘法后转FP32，要求逐位一致；不是相对O5/O6的量化MSE。

## 四样本×三轮诊断计时

warmup50、repeats200、单stream、native Driver CUDA Event、预分配、交错执行，
48条记录/9600个Event区间。ms为每样本跨轮median后再跨样本median；比值按同样本/同轮配对。
只测缓存单位scale计算核心，没有conversion/Cold/steady测试，也未扩大24样本。

|整数payload来源|通用v67，单位scale ms|单位scale专用 ms|诊断配对比值|CV≥3%，控制/专用（各12条）|
|---|---:|---:|---:|---|
|O7|0.449024|0.415744|1.076315×|6 / 8|
|O8|0.444928|0.411136|1.084478×|6 / 7|

共享、未锁频GPU，保留全部CV失败；没有bootstrap区间，不将波动归因于未经测量的外部活动。
48条输出均与单位scale reference逐位相同，`unit_reference_mse=0`；**没有新的原实验MSE**。

## 代码生成解释

下表为同一目标entry中完整整数K循环的静态指标，不是NCU动态耗时比例。

|指标|v67控制|单位scale专用|
|---|---:|---:|
|循环指令数|378|314|
|IMAD家族（包含寻址/其他整数指令）|154|91|
|标量LDS指令数|15|3|
|INT4 IMMA / LDSM|64 / 16|64 / 16|
|循环最大存活GPR|160|156|
|整个entry分配寄存器/线程|168|168|
|动态shared bytes / 驻留CTA|34304 / 3|34304 / 3|
|ptxas stack bytes|8|40|
|ptxas spill store/load bytes|8 / 8|48 / 48|
|循环local load|0|1条LDL.64|

专用入口并没有自动降低驻留寄存器预算或消除spill。每个目标entry的PTX/SASS均通过
两路原生INT4与cp.async/LDGSTS审计，无INT8替代；重编译控制的opcode计数和指令数与v67相同。
不能把完整entry的spill字节数直接当作热循环流量。

8项合成检查覆盖随机、全零、极值、正负交替及非默认stream，另验证非单位输入拒绝。
memcheck/synccheck均0 errors，racecheck为0 errors/0 warnings；范围M≤128、N≤256、K4096，
不是4096³ sanitizer验收。4096³单位scale输出由四样本计时运行另行核对。

## 结论

仅删除系数算术，在当前组织方式下仍约0.41ms，与必要MMA容量下界仍有明显差距。
这削弱了“继续局部改写scale乘法就可大幅接近下界”的依据；但不能证明scale只占约8%，
因为寄存器分配、调度、spill及计时噪声并非固定。下一步应依据真实依赖/供数证据挑选结构变化，
不再只以少乘法、零spill或更高occupancy作为收益证明。

真实数据最佳仍为v67 GEMM与v73全K在线准备；正式默认、O3和5090均未改变。
本轮不声称达到优化目标。

## 证据与复现

编译commit `a305e9f`，测试commit `738f6ac`。本地实现、GitHub push后A100 fetch/ff-only合入。
归档SHA256：`f968f10234f2f1d69de3f3e518577d02ef873e43002a74f5af36458bf9e74d2d`。
正式扩展SHA256保持`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
完整PTX/SASS、liveness、日志、source provenance和原始Events保留；cubin留在完整归档，不作Git源码依赖。

```bash
python scripts/probe_o78_unit_scale_codegen.py --output reports/v77_rebuild
python scripts/benchmark_o78_unit_scale_probe.py \
  --cubins reports/v77_rebuild --output runs/v77_recheck \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
python -m pytest tests/unit/test_o78_unit_scale_diagnostic.py tests/unit/test_roof_v77_evidence.py -q
```

输出目录必须全新；`--validate-only`用于有限sanitizer检查。

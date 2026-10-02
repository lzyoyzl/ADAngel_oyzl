# v57：扩大 M 复用 B，四样本初筛后淘汰

**结论：供数模型有利，但实际吞吐下降；不替换最佳54/59，不扩展24样本或端到端测试。**
本轮只测试一个候选，不扫描其它tile、stage或warp配置。

## 改动与选择理由

保持4个warp、N128、K128、N64寄存器流式片段、两路原生INT4、独立G128 scale和
原FP32累加顺序。仅将CTA从 `64×128×128` 扩为 `128×128×128`，
让每份B fragment在4个M atom间复用，而不是2个。O3仍3个cp.async阶段，O7/O8仍2个。
配置允许M128的头文件由原版作单一断言替换生成；原头文件、正式扩展、5090后端均不修改。

固定4096³，逻辑fragment供数量为：
`(2*M*64*2 + 128*64*2) * (4096/M) * (4096/128) * 32`。
两个2分别来自A低/高平面和warp分区复用；M翻倍使该量减少25%。
代价是每线程FP32 accumulator从64增至128。放宽launch bound到2CTA，仅为允许编译器
容纳新增寄存器，不强压到原168预算。这个候选不是严格只改变一个编译参数的微基准。

## 实测：4真实样本 × 3轮 Compute-only

样本为第一层q/k/v/o；4096³，50预热、200次CUDA Event测量，控制/候选同Driver、
同输入、循环换序。共72条记录、14,400个原始Event值。输入转换/分配在计时外。
表中ms为样本跨轮median再取样本median；吞吐变化由逐样本配对比值计算。

|后端|原M64 ms|M128 ms|配对吞吐变化|加速比描述性95% CI|CV≥3%：原/新，各12条|
|---|---:|---:|---:|---|---:|
|O3|0.451328|0.476672|**−5.16%**|[0.94563,0.95207]|4/3|
|O7|0.487424|0.536576|**−9.08%**|[0.90613,0.92982]|9/2|
|O8|0.486400|0.530944|**−8.44%**|[0.90643,0.91667]|7/1|

未锁频，全部离群和CV失败保留；不能称为严格稳定性验收通过。
区间仅描述这4个同trace样本，不是24样本或模型总体结论。

|输出MSE / FP16参考，仅这4样本|Median，原/新相同|Mean，原/新相同|
|---|---:|---:|
|O3 / O0|0.000278282719670|0.000283055340644|
|O7 / O5|0.000102067943873|0.000099167494159|
|O8 / O6|0.000123493256717|0.000122317650852|

全部已测输出finite FP32、与当前最佳逐位相同；新旧输出MSE为0。
这些四样本MSE不能与此前24样本的median直接比较。没有新增conversion、Cold或steady测量。

## 审计、资源与解释

|指标|M64|M128|
|---|---:|---:|
|寄存器/线程|168|255|
|查询的驻留CTA上限 / 计算warp上限|3 / 12|2 / 8|
|Shared bytes，O3 / O78|50688 / 34304|75264 / 51200|
|ptxas spill store/load bytes，O3|12 / 12|16 / 16|
|ptxas spill store/load bytes，O78|8 / 8|16 / 16|
|每个展开G128主体静态LDSM条数|16|24|
|每个展开G128主体静态IMMA / I2F / FFMA，各自|64|128|

CTA总数减半，LDSM模型工作量下降25%，但MMA/I2F/FMA总数学工作不减少。
寄存器占用增大、可驻留warp减少，会削弱延迟隐藏；实测表明供数复用收益不足以抵消代价。
这是资源与代码支持的解释，本轮未追加NCU，不能据此定量归因到某个stall百分比。
静态opcode、逻辑请求量不冒充实测HBM流量或NCU动态计数。

同一PTX entry与SASS function确认U4×S4、S4×S4、cp.async.cg/LDGSTS.BYPASS，无INT8替代。
M64控制与已有54/59控制完整编码SASS相同。96项合成检查覆盖4形状、4pattern、3后端、
两配置，K128/384/640/4096、随机/零/极值/零scale与非默认stream；既核对FP64语义参考，
也强制逐位等于最佳。memcheck重跑96项，0 errors。没有新synccheck/racecheck结果。
memcheck最大K4096形状为128×128；完整4096³来自四样本数值测试，不混淆覆盖范围。
最初一次memcheck用了无效过滤语法，未运行测试；其日志保留，正确结果为`memcheck_valid.log`。
归档后另有6项CPU测试通过：复算全部原始Event统计与配对汇总，核对资源、MSE和安全检查覆盖范围；
这些归档回归检查不算作新增GPU测试。

## 复现与保存位置

实现commit `b041f64`；Driver/测试commit `12bfbef`。先本地实现并push，再A100 fetch/ff-only merge。
原生扩展SHA未变：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
PYTHONPATH=python python scripts/probe_roof_m128_codegen.py --output reports/v57_rebuild
PYTHONPATH=python python scripts/validate_roof_m128_probe.py \
  --cubins reports/v57_rebuild --output reports/v57_recheck
PYTHONPATH=python python scripts/benchmark_roof_m128_probe.py \
  --cubins reports/v57_rebuild --output runs/v57_recheck --samples 4 --rounds 3 --warmup 50 --repeats 200
```

目录必须新建。memcheck采用`--kernel-name kns=adangel_roof_m128_ --error-exitcode 86`，运行相同验证脚本。
[原始结果与汇总](runs/o378_roof_v57_screen/summary.json)、
[完整审计](reports/o378_roof_v57/codegen.json)、[内存检查](reports/o378_roof_v57/memcheck_valid.log)均保留。
归档SHA256：`e822f6556ab5ae5899c46f923e8072328ce4aab60280b4545f7c9308c5a4bda6`；cubin和.so留在A100，不上传Git。

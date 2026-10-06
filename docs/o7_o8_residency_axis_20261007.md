# v107：A/B 驻留轴交换未通过潜力门槛，停止候选

**结论：没有新的性能提升或 MSE 结果，当前最佳和正式默认不变。**
只新增一个此前未测试的供数结构，不扫描参数；完成 A100 编译及指令审计后停止。

只测试一个未做过的 GEMM 结构，不扫描 tile/stage/warp/链数。
CTA64×128×128、4 warp、2-stage、G128 factor/全K范围guard、FP32输出不变。

```text
v78：A(M64)常驻32寄存器；B按N64流入16寄存器；每窗口2M×4N=8条MMA链
v107：B(N128)常驻32寄存器；A按M32流入16寄存器；每窗口1M×8N=8条MMA链
```

两种窗口每线程均32个INT32 partial、64个最终accumulator、48个逻辑操作数寄存器。
编译确认必要MMA64、LDSM16/G128和global payload copy不变；改变的是供数/生命周期关系，
不是降低量化精度，也不把源码生命周期变化直接当作硬件加速。

查重：v48保留完整N128时仍同时保留两个M atom并使用独立high/low partial；
v94仅4链并增加A读取至24 LDSM；v88只交换原窗口内的遍历顺序；v49增加B lookahead但机器码不变。
本候选保持8链及32 partial，真正交换驻留轴，不重测上述方向。

先用CuTe实际partition_A/B/C验证128线程、每个K64/M32、全部8192输出唯一owner。
正式entry要求同一PTX/SASS含原生U4/S4与S4/S4和cg copy；旧v78编码必须不变。
测试前规定潜力gate：regs≤原值、热循环无local、MMA/LDSM不增加，并且静态指令减少≥5%，
或主循环最大live GPR至少减少16，或自然分配达到≤128regs。后两项不是性能/occupancy保证。
gate失败即停止，不做候选GPU性能/MSE/NCU；通过后才考虑完整24样本配对验收。

## 实际编译结果

|指标|v78 对照|v107 候选|
|---|---:|---:|
|整数主循环静态指令|383|379（−1.04%，不是实测加速）|
|主循环最大活跃 GPR|166|166|
|整个 entry 分配寄存器/线程|168|168|
|每 G128 原生 S4/S4 + U4/S4 MMA|32 + 32|32 + 32|
|每 G128 LDSM|16|16|
|每 G128 异步 global→shared copy 指令|10|10|
|整数主循环 LDL/STL|0 / 0|0 / 0|
|整个 entry stack / spill load / spill store B|0 / 0 / 0|0 / 0 / 0|
|FP32 回退主循环静态指令|440|453|
|预设潜力 gate|对照，不适用|未通过|

同一正式候选 entry 的 PTX/SASS 确认原生两路 INT4 与 cg copy，没有 INT8 替代。
重新编译的 v78 对照指令编码与原 v78 完全一致。源码中的回退路径未修改，
但候选整个 entry 的编译布局使其回退 SASS 改变；没有 GPU 数值测试，不能声称逐位一致。

**停止原因：**寄存器压力和必要 MMA/供数工作没有减少，仅少4条主循环静态指令，
低于事先规定的5%工作量或16个活跃GPR下降门槛。没有放宽门槛去寻找计时噪声中的收益。
这只说明此候选不值得在当前额度下晋级，不证明它一定更慢，也不否定所有供数优化。

不运行候选 GEMM、24样本性能/MSE、NCU或sanitizer，不用历史 MSE 代替本轮正确性。
O3 v89、O7/O8 v78+v73 仍为主基准；既有 v99/v106 独立微调的结论不变。
只在独立 cubin 中编译，不重新编译正式扩展，不改默认、conversion、O3或5090。

## 修复和复核边界

首次检查拒绝 A/B 不同 tensor 类型的合并 `auto` 声明；第二次坐标检查发现
完整 M64 TiledMMA 不能直接 partition M32 切片。改用同 warp 布局的 M32 slice MMA，
逐值与原 M64 坐标对照；第三次拒绝 signed/unsigned slice 的合并 `auto` 声明。
修复后，128线程、49152次 A/B 坐标比较及8192输出唯一 owner 全部通过。
这四次是同一候选的编译检查/修复，不是四轮优化或 GPU 性能测试。所有失败日志保留。

构建 commit：`e44142d9a4feddc247385c71a2cf8d8817a4fd80`；CUDA12.8.93、pinned CUTLASS。
先本地实现并推送 GitHub，再在 A100 仓库 fetch/ff-only merge。正式 A100 扩展 SHA256
仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
[源码/指令/资源/失败日志与复算证据](evidence/a100_o378_roof_v107/README.md)。

```bash
python -m pytest tests/unit/test_residency_axis_codegen.py tests/unit/test_roof_v107_evidence.py -q
# 如需复现编译，输出目录必须不存在；不是性能测试。
python scripts/probe_residency_axis_codegen.py --output reports/v107_compile_recheck
```

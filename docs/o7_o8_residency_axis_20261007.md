# v107：固定预算的 A/B 驻留轴交换，先编译筛选

只测试一个未做过的 GEMM 结构，不扫描 tile/stage/warp/链数。
CTA64×128×128、4 warp、2-stage、G128 factor/全K范围guard、FP32输出不变。

```text
v78：A(M64)常驻32寄存器；B按N64流入16寄存器；每窗口2M×4N=8条MMA链
v107：B(N128)常驻32寄存器；A按M32流入16寄存器；每窗口1M×8N=8条MMA链
```

两种窗口每线程均32个INT32 partial、64个最终accumulator、48个逻辑操作数寄存器。
必要MMA64、LDSM16/G128和global payload copy预计不变；改变的是供数/生命周期关系，
不是降低量化精度，也不把源码生命周期变化直接当作硬件加速。

查重：v48保留完整N128时仍同时保留两个M atom并使用独立high/low partial；
v94仅4链并增加A读取至24 LDSM；v88只交换原窗口内的遍历顺序；v49增加B lookahead但机器码不变。
本候选保持8链及32 partial，真正交换驻留轴，不重测上述方向。

先用CuTe实际partition_A/B/C验证128线程、每个K64/M32、全部8192输出唯一owner。
正式entry要求同一PTX/SASS含原生U4/S4与S4/S4和cg copy；旧v78编码必须不变。
预先规定潜力gate：regs≤原值、热循环无local、MMA/LDSM不增加，并且静态指令减少≥5%，
或主循环最大live GPR至少减少16，或自然分配达到≤128regs。后两项不是性能/occupancy保证。
gate失败即停止，不做候选GPU性能/MSE/NCU；通过后才考虑完整24样本配对验收。

只在独立cubin中编译，不重新编译正式扩展，不改默认、conversion、O3或5090。
状态：本地实现/CPU契约阶段，尚无编译后资源、性能或MSE结论。

```bash
python -m pytest tests/unit/test_residency_axis_codegen.py -q
python scripts/probe_residency_axis_codegen.py --output reports/o378_roof_v107_codegen
```

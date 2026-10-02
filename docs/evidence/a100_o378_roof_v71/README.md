# v71：O7 幂二移位候选在编译门槛停止

结论：**不进入GPU性能初筛，不扩展24样本，不改默认或当前最佳。**
这是有完整编译/坐标/整数算术证据的提前筛选，不是一次测得负加速的GPU实验；
没有新增延迟、加速百分比或MSE数值。

## 尝试了什么

v70表明当前全K整数路径仍有大量IMAD与发射/依赖开销。
O7的UE8M0激活scale相对行锚点一定是`A_factor=2^d`，因此本轮只尝试：

```text
控制：acc += partial * (A_factor * W_factor)
候选：acc += bitcast_int32(uint32(partial * W_factor) << d)
```

沿用v69的系数、乘积、任意前缀与FP32 epilogue安全检查，以及原FP32回退；
接受路径的`0<=d<=30`且A_factor非零，原界也保证`partial*W_factor`不会溢出。
移位使用无符号位模式，避免负整数左移的未定义行为，最后只是位解释，不做浮点运算。
它不改变两路原生INT4、量化、G128独立scale或整数累加顺序。
O8的激活scale含尾数，不适用本特化，未修改。

每线程只计算4个唯一行的移位量，通过CuTe输出坐标复用到64个accumulator。
host坐标程序验证全部128线程、8192个输出的行映射与唯一覆盖；
CPU测试覆盖全部31种允许移位、多种权重因子、正负边界与随机整数，27项既有/新测试通过。
**这些不能替代GPU正确性与MSE测试，本轮未运行候选GPU kernel。**

## 编译审计发现

A100 SM80、CUDA12.8、固定CUTLASS commit；本地源码先push，A100 fetch/ff-only到`d497c3d`编译。
控制直接使用未修改的v67 full-K函数，两份代码保留相同的分支与回退；
控制整入口指令数和opcode计数均与v67一致。正式扩展未重新编译。

|静态指标|原full-K控制|幂二移位候选|
|---|---:|---:|
|寄存器/线程|168|168|
|编译报告spill load/store字节（各）|8|24|
|stack frame B|8|16|
|主循环指令条数|378|453|
|主循环原生U4×S4 IMMA|32|32|
|主循环原生S4×S4 IMMA|32|32|
|主循环IMAD族（含IADD等变体）|154|141|
|主循环SHF.L.U32|2|65|
|主循环FLO.U32|0|4|
|主循环local load条数|0|4|
|主循环LDSM|16|16|

表中为**单个循环体的静态指令**，不冒充NCU动态数量或耗时比例。
解析器通过回跳分支和64条INT4 MMA定位整数主循环，并排除FP32回退；不按人工行号截取。
主循环总工作约增加19.84%，必要MMA和shared fragment加载没有减少。

关键原因是**移位没有免费替代整条乘加**：原代码可用IMAD完成乘法并累加，
新代码需要权重乘法、可变移位、再相加；其中32条相加又被编译成`IMAD.IADD`。
因此不能把源码里去掉一次乘法解释为每输出少一次数学管线工作。
新增4条LDL确实位于整数循环的预取分支中，不像v70的8 B静态spill仅在回退路径。
这仍是控制流/静态证据，没有采集本候选的动态local流量。

## 决策与后续方向

两路原生INT4和异步拷贝审计通过，但本候选没有足够的资源改善依据：
数学工作仅部分转移，发射工作增多，循环新增local读取。
为避免在低希望方向消耗测试机会，停止，不进行相邻表达式/寄存器参数扫描。
这**不证明实测一定变慢**，也不证明所有幂二特化均无效；只说明当前写法不值得扩大测试。

下一步不再仅替换乘法语法，应优先寻找能减少必要后处理数量、缩短真实依赖，
或减少fragment供数的结构性差异，并先核对已失败候选，避免重复tile/warp/cache搜索。
最新普通Event、四模式和MSE仍引用v67/v69；O3、O8、5090和正式默认均保持不变。

## 可复核证据

- [编译与源码指纹](reports/o378_roof_v71_codegen/codegen.json)
- [循环体计数及local指令地址](reports/o378_roof_v71_codegen/loop_analysis.json)
- [完整SASS](reports/o378_roof_v71_codegen/o7_pow2.sass)
- [ptxas资源报告](reports/o378_roof_v71_codegen/build.log)
- [CuTe坐标检查](reports/o378_roof_v71_codegen/coordinates.log)

归档SHA256：`6087736627ca4611603dec53c8fe2dfd8db4a5ccca625c397a5b67d8a719aedf`。
完成后已复核正式扩展SHA仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
加上归档复算以及v70/旧NCU解析回归，本轮共55项本地测试通过。

```bash
python scripts/probe_o7_pow2_codegen.py --output reports/v71_recompile
python scripts/analyze_o7_pow2_codegen.py \
  --sass reports/v71_recompile/o7_pow2.sass
python -m pytest tests/unit/test_o7_pow2_candidate.py \
  tests/unit/test_o78_fullk_integer_metadata.py -q
```

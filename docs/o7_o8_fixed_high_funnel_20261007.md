# O7/O8 固定 high×16 的指令路由检查（v117）

状态：编译门槛通过，GPU正确性与24样本配对完成；性能负向，不采纳，不改正式默认或当前最佳。

## 为什么不是重复优化

v71处理可变scale与partial的移位/加法，v87把可变激活因子改为CLZ＋系数移位，结果更慢。
本次只改变两路INT4合成中的固定high×16：量化、系数形成、整数累加顺序、八条MMA链、
32个partial/64个累加寄存器、CTA64×128×128、四warp/两stage和转换/epilogue均不变。
不重新测试tile、stage、producer、partial存储或累加链数。

最佳v78的整数循环383条指令，IMAD族158条（含地址运算），固定high重构混用IMAD.SHL与SHF。
实际def-use核对为21条IMAD.SHL与43条SHF，不把地址移位计入64个high分量。
最佳O8的预热NCU中FMA/ALU管线活跃比例约37.96%/21.65%，提示可能存在路由空间，
但这些百分比不是可相加的延迟，也不能据此推算加速。

## 精确变换

使用公开PTX的两源funnel shift：d=(b<<4)|(a>>28)。取b为high partial的原始32位，
a为threadIdx.x。实际CTA128线程，甚至SM80合法CTA最大1024线程时，a>>28恒为0。
G128 signed-high partial绝对值不超过8192，重构×16不溢出INT32；无新的舍入、定点误差或signed-shift UB。
非零下字只为避免编译器将简单左移重新选成IMAD.SHL；不修改SASS或CUDA/CUTLASS系统库。

[NVIDIA PTX 12.8：shf定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#logic-and-shift-instructions-shf)。
[NVIDIA开发者对ALU/FMA指令路由的说明](https://forums.developer.nvidia.com/t/pipeline-operator-forwarding-for-integer-instructions-in-cuda/299761/12)
只作为路由假设，不作为本kernel的SM80延迟模型。

## 预先固定的投入门槛

1. 新旧控制kernel机器码一致；同entry保留两路原生INT4，每G各32条MMA。
2. 从第二次signed MMA追踪到第一次unsigned MMA，64个high分量全部由SHF重构，排除地址移位误计。
3. 分配寄存器不超过168、整数热循环LDL/STL为0；LDSM16与异步copy10不变。
4. 总循环指令最多增加1%，IMAD族指令至少减少10%。这是成本筛选，不是预测性能。

门槛失败就停止，不扫描seed/opcode相邻变体。通过后先核对synthetic输出、guard/fallback，
再直接进行24样本三轮交错A/B、逐位输出/MSE与计时对照；不做小规模性能初筛。
只有实测收益成立才继续四种模式、GPU安全和必要NCU验收。O3暂不迁移该路由。

## 编译检查结果

同一CUDA12.8/CUTLASS固定commit下，候选循环385条（+0.522%），IMAD族137条（−13.291%）。
64个high分量全部走SHF，两个原生INT4各32条、LDSM16、copy10不变；168regs，整数热local为0。
旧v78控制编码完全一致，正式扩展SHA不变。不能将13.291%指令路由减少称为性能提升。

运行（所有路径在A100项目内）：

```bash
python scripts/benchmark_o78_high_funnel.py \
  --cubins reports/o378_roof_v117_codegen \
  --output reports/o378_roof_v117_compute24 \
  --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
```

脚本先运行synthetic正确性与guard/fallback验证，再做24样本交错配对；禁止4样本初筛。

## 完整24样本结果：停止该路线

同进程/同输入/单stream、三轮交错，warmup1000、每轮200次，原始记录全部保留。
下表为每样本三轮median的跨24样本median。配对加速比先逐样本/逐轮比较，
因此不强制等于两列跨样本median的商；CI由24个样本配对比bootstrap获得。

| 后端 | 原v78 GEMM ms | v117 GEMM ms | 配对吞吐变化 | 配对加速比95% CI | CV≥3%旧/新 |
|---|---:|---:|---:|---|---:|
| O7 | 0.475648 | 0.482304 | −1.469% | [0.985138, 0.986577] | 1/72、0/72 |
| O8 | 0.481280 | 0.487936 | −1.468% | [0.985138, 0.985325] | 1/72、1/72 |

64项synthetic输出与12项边界检查通过；全部288条真实配对记录finite FP32、metadata正确，
相对原v67和旧v78输出逐位一致。O7全部CTA走安全整数路径；O8的layer_24_o_proj为2036个整数CTA、
12个原有FP32 scale fallback CTA，其余样本均2048个整数CTA。新旧guard/fallback逐记录完全一致。

| MSE主参考 | 旧/新 median MSE | 旧/新 mean MSE | 新旧输出差MSE |
|---|---:|---:|---:|
| O7 / O5 | 0.005536172273439442 | 0.005053635851002639 | 0 |
| O8 / O6 | 0.004411084910985704 | 0.00438137929907354 | 0 |

这不是新的conversion、Cold或steady-state测量：v73准备完全复用，GEMM已明确变慢，
无需继续端到端或NCU重复采集；未对新候选执行sanitizer，不宣称完成正式GPU安全验收。
少量CV离群保留、不删除，没有锁频或等待GPU空闲。

## 得到的结论

确定的成本变化是：IMAD族158→137，总循环383→385，S2R5→7；MMA、LDSM、copy数量和资源不变。
静态def-use中已开始尚未结束的MMA链峰值8→6，只描述机器码顺序，不是实测并发度。
因此“把FMA工作搬到ALU”同时改变了供数/指令调度，并没有消除MMA或scale数学工作。
实测说明这一固定路由不是当前最佳内核的有效优化点；不能仅凭这些静态数据断言
全部负收益由S2R或链峰值单独造成，因果分解需要额外诊断，本轮不投入重复扫描。

当前最佳继续为O3 v89、O7/O8 v78＋v73。停止本候选，不试相近seed/opcode变体，
不迁移O3；正式扩展、量化、trace、转换和5090均未修改。
[原始完整配对与编译证据](evidence/a100_o378_roof_v117/README.md)。

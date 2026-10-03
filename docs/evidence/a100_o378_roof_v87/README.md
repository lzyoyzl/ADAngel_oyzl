# v87：O7 先移位生成系数，保留整数乘加；初筛未获益

**候选停止，不切换正式默认。** 本轮只测试 O7，O3/O8/5090 不变。
与同轮 v78 配对，首层四样本三轮 GEMM 吞吐下降 **3.05%**；输出逐位一致，MSE 不变。
不扩大24样本或四模式性能测量，不将通过编译审计当成加速。

## 数学与实现

O7 的 UE8M0 激活 scale 在原全K精确分解后，组因子 `A_factor=2^d`。
原有范围保护保证被接受的行 `0<=d<=30`，且系数、乘积和每个整数累加前缀均不溢出 INT32。

```text
原路径：coefficient = A_factor * W_factor
        accumulator += partial * coefficient

候选：  d = log2(A_factor)             // 每线程4个不同row因子
        coefficient = W_factor << d   // 无符号PTX shf.l.wrap.b32
        accumulator += partial * coefficient
```

只改系数生成，不移位负整数 partial；最后一步仍编译为整数 IMAD。
与 v71 的“先乘 partial、再移位、再相加”不同；与 v72 的系数乘法也不同。
保持 v78 八条独立 MMA 链、CTA64×128×128、4warps、两stage、两路原生INT4、
cp.async、独立G128 scale、整数guard和原FP32回退、最终FP32 epilogue 不变。
同一 v73 在线转换/metadata库，**没有新布局或免费离线准备**。
O8 的激活scale还含尾数，不能假定因子为2的幂；驱动在任何GPU调用前拒绝O8。

## 编译审计

固定 CUDA12.8.93、SM80、现有 CUTLASS commit。新cubin中的v78控制入口编码SASS
与原v78逐条相同；计时控制直接加载原v78 cubin，而非用重编译入口替换控制。
对同一实际候选entry做PTX/SASS与寄存器定义使用追踪：64个系数SHF均由scale生成，
64个后续IMAD均保留 `partial*coefficient+acc`；两个入口均为原生U4×S4和S4×S4，无INT8替代。

以下为每个G128整数主循环的**静态**计数，不是NCU动态测量。

|指标|v78控制|v87候选|
|---|---:|---:|
|寄存器/线程；最大CTA/SM|168；3|168；3|
|动态shared/CTA bytes|34304|34304|
|主循环静态指令|383|380|
|原生S4/U4 IMMA|32 / 32|32 / 32|
|LDSM.16.M88.4|16|16|
|LDGSTS.128|10|10|
|普通IMAD|128|64|
|全部IMAD前缀指令|158|117|
|系数SHF.L.W.U32.HI|0|64|
|FLO.U32|0|4|
|入口stack / spill load / spill store bytes|0 / 0 / 0|0 / 0 / 0|
|主循环LDL / STL|0 / 0|0 / 0|

64次普通IMAD减少不等于所有整数工作减少64次：SHF、FLO及地址相关IMAD等同步变化，
总静态指令仅少3条。必要MMA、供数与驻留数都未改善。
本轮未新增NCU，**不能据此定量断言某条管线或某个stall贡献了多少退化**。

## 四样本三轮配对结果

A100，4096³，`layer_00_{q,k,v,o}_proj`；warmup50、repeats200、inner100。
原生CUDA Event、单stream、预分配、交错控制/候选顺序。每条结果保留200个原始事件时间。
这里只报告cached compute-only；两者准备方法相同，但不以它推算新的Cold/steady成绩。

|指标|v78控制|v87候选|
|---|---:|---:|
|样本内跨轮median再跨样本median ms|0.448512|0.461312|
|同样本配对吞吐变化|基准|**−3.05%**|
|配对speedup bootstrap 95% CI|—|[0.962054, 0.990741]|
|选定指标CV≥3%的记录/12|12 / 12|10 / 12|
|O7/O5 输出MSE median|0.000102067943157889|相同|
|O7/O5 输出MSE mean|0.000099167492984850|相同|
|相对原v67输出差MSE / 最大绝对差|0 / 0|0 / 0|

配对比值先在同一样本内计算再汇总，不等于上面两列总median直接相除。
区间仅描述这四个相关trace样本，不外推为24样本或独立模型总体。
24条记录全部finite FP32、metadata精确匹配；每条均2048个整数CTA、无回退/非法CTA。
两种kernel都对同一个v67结果逐位一致，因此它们彼此也逐位一致。
MSE较24样本汇总小是**样本集合不同**，不是精度提高。

共享未锁频GPU，保留所有离群及CV失败；未通过严格稳定性门槛。
不把GPU快照当作每个离群事件的原因证据，不筛掉慢记录以构造收益。

## 正确性和安全性范围

32项数值检查覆盖O7的随机/全零/正负交替/宽scale、两种实现、四种计时模式，
含非默认stream、finite FP32、FP64语义参考容差及对v67逐位一致。
9项额外边界覆盖非法UE8M0/E4M3、factor范围、范数饱和、epilogue上/下溢和回退。
另外用单个非零乘积穷举31个合法 `d=0..30`，控制/候选/v67共93次输出均与解析结果逐位一致，
全部走整数路径，含正负号；并检查O8在启动GPU前被拒绝。
这些覆盖M/N≤128/256、K=4096，**不是4096³全输入的内存安全证明**。
memcheck、synccheck各0 errors；racecheck为0 errors / 0 warnings。
三种工具均完成上述32+9项检查和31个移位解析对照，原始日志与validation.json全部保留。
本地57项相关单元/证据测试通过，包含从24条原始Event重算统计、完整配对汇总、
MSE不变、控制机器码身份、同entry原生INT4和系数定义使用链核对。

## 结论与复现

这次实验否定了“将系数乘法换成移位、并保留最终IMAD就能提速”的当前实现。
减少某类算术指令没有缩短实测时间；不继续枚举相邻移位写法，不扩大负收益候选。
当前最佳仍O3 v79、O7/O8 v78配合v73准备；约0.220347ms的必要MMA容量下界未变，
它不是对真实依赖图可达延迟的承诺，优化目标仍未达成。

编译commit `c50d508`；运行驱动commit `f9d0e36`。均先本地实现和GitHub push，再A100 fetch/ff-only merge。
候选cubin SHA256：`ec6d0a011dfa1f956d495fb8b69016195155ed4569ddfa54b65ee6e82acd4255`。
正式扩展SHA256仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
编译证据下载归档SHA256：`6e88b4db41b5b68d0f1f1a7f5b0d91aed2a74c901a5c6e811d0d3917a0155bf2`。
完整编译/计时/安全证据下载归档SHA256：`c286898c3bef9429da2ca5ad43e006c4d8f70fff9b97b9274582abec8912d618`。

```bash
python scripts/probe_o7_shift_coefficient_codegen.py --output reports/o378_roof_v87_rebuild
python scripts/benchmark_o7_shift_coefficient.py \
  --cubins reports/o378_roof_v87_rebuild --output runs/o378_roof_v87_recheck \
  --samples 4 --rounds 3 --warmup 50 --repeats 200 --inner 100
```

需保留现有v67、v78 cubin和v73准备库；输出目录须不存在。没有重编译或切换正式扩展。
`runs/o378_roof_v87_screen/` 保存原始Event、summary、GPU快照、源数据provenance和环境。
`reports/o378_roof_v87_codegen/` 保存生成头、PTX、SASS、资源/存活/坐标检查与构建日志。
二进制cubin/坐标验证程序留在本地与A100归档，不提交Git。

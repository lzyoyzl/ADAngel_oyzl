# O3 两路 INT4：补齐一次未运行候选的实测（v135）

**结论：不采用，当前最佳不变。** 完整24样本三轮配对，DP2A候选吞吐下降6.48%；
输出与旧最佳逐位一致，MSE不变。此路线结束，不扫描邻近参数或迁移O7/O8。

## 实测结果

固定4096³，同进程/同输入/同stream交错执行，warmup1000、repeats200、inner100。
不锁频、不等GPU空闲，保留全部144条记录和57600个GEMM/total Event时长。

| 指标 | 旧最佳 O3 v89 | 冻结 v115 DP2A候选 |
|---|---:|---:|
| Compute-only GEMM median ms | 0.435712 | 0.467200 |
| 配对吞吐 / 旧最佳 | 1.0000× | 0.93524×（−6.48%） |
| 配对速度比 bootstrap 95% CI | — | [0.93085, 0.93873] |
| CV≥3%的记录 | 2/72 | 0/72 |
| 输出 MSE vs O0，median | 0.00665301028741 | 0.00665301028741 |
| 输出 MSE vs O0，mean | 0.00757884701330 | 0.00757884701330 |
| 每线程寄存器 / 实际CTA驻留数 | 168 / 3 | 168 / 3 |

延迟为每个样本先取三轮中位数，再对24样本取中位数；速度比先做同样本/同轮配对再汇总，
因此不能简单用两列全局延迟相除来替代配对结果。三轮分别下降6.33%、6.27%、6.37%，
方向一致；全部保留，不因旧版2条CV超标而挑选/删除记录，也不为明显负收益追加重测。

候选实际DP2A路径覆盖平均80.73%的CTA，其余保留原整数/浮点回退，不能说所有CTA都走DP2A。
24样本输出、packed payload及scale均与旧最佳逐位一致；新旧输出MSE严格为0。
另通过96项合成验证、8项非法输入拒绝、2项grouped坐标及4项三路径同launch验证。
这些是数值/接口验收；本轮没有新NCU或Compute Sanitizer，不冒充完整内存安全验收。

## 为什么没有采用

高低路独立只能减少一类依赖，不能免除实际工作：每G128仍有32+32条原生INT4 MMA，
还需要64条PRMT打包和64条标量DP2A；总静态指令323→340，并增加3条热local读取。
没有换来更高驻留容量。实测说明这组取舍不合算，但没有新NCU，**不能把6.48%的退化全部归因于spill或某一条指令**。

不扩展正式conversion/Cold/steady测量，不把GEMM差值当成端到端差值。
运行器已将额外metadata打包计入W转换/Cold路径；compute-only中提前准备，这是与原口径一致的缓存语义。
O3 v89、O7/O8 v78的最佳GEMM保持；正式扩展SHA、默认、5090均未改变。

## 方法和历史筛选的区别

本轮不是新的 CUDA 优化，也不是重跑已有负性能结果：复用 v115 的原始 cubin，
首次测量其完整24样本性能。O7/O8、正式默认和5090不变，不引入单路 INT8 Tensor Core。

旧 v115 将 high/low 两路原生 INT4 点积分开，再使用 PRMT+DP2A 进行精确整数合并：
`acc += low_dot*f + high_dot*(16*f)`。DP2A 是普通整数指令，不是 INT8 Tensor Core。
独立点积缩短合并前的依赖，但增加 packing、地址工作以及3条热 local 读取。
原编译 gate（340/323条静态指令、3条local）仍明确为失败，不覆盖或改写旧记录。

此前的5%工作量门槛是投入筛选，不是性能定理。依据用户已允许少量spill的条件，
这次只对**相同冻结二进制**做一次运行复核，不调整寄存器、布局或邻近参数。
保留原始scale、整数范围guard及两种回退；只有原guard安全且factor≤15的N128 CTA使用DP2A。
实际覆盖率已由GPU metadata与独立CPU packing对照重新核对。

流程：原始PTX/SASS与SHA回放 → 实际3 CTA/SM容量 → 数值/边界/三路径混合验证 →
24样本×3轮、warmup1000、repeats200的交错配对、MSE及全部CV。
compute-only有明确收益才扩四模式；新增metadata打包计入W转换/Cold，compute和steady缓存。
无收益即结束这条实测路线。不能把静态指令变化称为加速，也不能将编译通过当作正确性通过。

```bash
TMPDIR="$PWD/tmp" python scripts/benchmark_o3_dp2a.py \
  --output runs/o378_roof_v135_reproduce --samples 24 \
  --rounds 3 --warmup 1000 --repeats 200 --inner 100 --modes compute_only
```

## 验证脚本修复与证据

首次运行确认两端实际均为3 CTA/SM。验证脚本的旧计时断言只接受两个W准备kernel，
在新增metadata打包（实际第三个kernel）时终止；尚未进入真实样本性能测试。
修复仅更新本轮计时契约检查，保留第三个kernel的真实开销与首次失败日志，不改CUDA二进制。
第二次运行通过原验证集后，新增三路径测试给FP64语义参考传入了三维plane视图，
而参考接口需要二维`[2*M,K/2]`；修正测试端reshape。两次均未进入24样本性能计时，
不是两轮优化结果；GPU数学及冻结cubin仍未改变。

第三次完整运行成功（源码`c47fd55`）。两份初始失败日志、原始Event、输入SHA、资源及统计重算结果
见[完整证据](evidence/a100_o378_roof_v135/README.md)。候选PTX/SASS复用[v115原始审计](evidence/a100_o378_roof_v115/README.md)，
核对同一cubin SHA及原生S4×S4/U4×S4，无INT8 Tensor Core。旧compile gate保持失败记录。

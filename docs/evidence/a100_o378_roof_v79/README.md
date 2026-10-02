# v79：O3 八条独立 INT4 累加链合并

**结论：小幅改善，保留独立候选，正式默认不变。**
24样本×3轮，O3 cached GEMM从0.456704降到0.447488ms，配对吞吐约+1.83%。
另一次24样本四模式中Cold约+1.09%；steady约+1.72%，但95%区间跨1，尚未确认收益。
转换代码完全相同，无确认变化。360条结果输出对v61逐位一致，MSE不变。
仍有CV失败，不是严格稳定性验收；没有达到约0.220347ms的理想容量下界。

## 实现与比较范围

只将v78已测的八链MMA组织方式迁移到O3；不能把O7/O8的收益外推为O3的结果。
控制为v61安全全K整数累加，候选为`adangel_roof_o3_eight_chain_candidate`。
两者均使用相同conversion2和GPU factor/guard，不拿“不带guard”的旧54路径冒充同成本对照。

```text
每个N64 slice，同时处理两个M atom，保留八条独立partial链：
  partial = high_K0 × W_K0 + high_K1 × W_K1
  partial *= 16
  partial = MMA(low_K0, W_K0, partial)
  partial = MMA(low_K1, W_K1, partial)
  integer_acc += partial × W_factor[column, group]
最后恢复原W anchor与A row scale，输出FP32。
```

INT8 high/low分解、Q4映射、逐G128 scale语义、full-K整数安全界及不安全CTA的原FP32回退不变。
high乘16中间值有界，使用整数乘法而不是负有符号整数左移。
CTA64×128×128、4warps、三阶段cp.async、50688B shared、最终单次输出写回不变。
不改O7/O8、RTX5090、正式扩展或正式默认。

## 同entry指令与资源审计

|指标|v61控制|v79候选|
|---|---:|---:|
|REG/线程；运行时驻留CTA/SM|168；3|168；3|
|整数主循环最大存活GPR|158|162|
|整个entry stack / spill stores / spill loads B|32 / 52 / 52|24 / 28 / 28|
|整数主循环静态指令|315|319|
|整数主循环静态LDL / STL|7 / 0|4 / 0|
|U4×S4 / S4×S4原生IMMA|32 / 32|32 / 32|
|LDSM / cp.async对应LDGSTS|16 / 9|16 / 9|
|U4 MMA的零C输入条数|16|0|

最后一行证明候选的low MMA确实接收已移位的high partial。
PTX/SASS同一entry包含原生两路INT4及cp.async/LDGSTS，没有INT8替代。
组合编译中的v61控制及O7/O8 sentinel与原v61编码SASS完全相同。
候选不是零spill；静态local load减少不能直接解释为某一百分比的stall/时间减少。
本轮未新增NCU，不从静态计数虚构动态瓶颈占比。

## 计时口径

A100、4096³、原24个真实trace，50次预热、200次测量、conversion inner=100。
单CUDA stream，buffer预分配，控制/候选交错执行，保留所有离群和CV失败。
两条路径W转换均为“转换+GPU factor/guard”两个kernel；A转换均一个kernel。
compute预备全部输入；steady缓存W和guard；Cold包含在线A/W准备。
公共原始FP16到INT8/MXFP4源格式的准备不计入本轮转换；没有新量化规则。

转换用批量摊销，conversion total为每对阶段样本之和；GEMM和端到端total由单次CUDA Event直接测量。
每轮ms为样本内跨轮median后跨样本median；吞吐先算同样本配对比值，不等于两列总median直接相除。
bootstrap区间是24个相关trace样本的描述性比较，不是所有负载的保证。
三轮GEMM与单轮四模式是不同run，不能拼成一套同时测得的最优时间。

### GEMM-only

|范围|控制ms|候选ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|首层4样本×3轮初筛|0.436224|0.429824|+1.53%|[1.006002,1.034063]|
|24样本×3轮确认|0.456704|0.447488|**+1.83%**|[1.013636,1.020606]|
|独立24样本四模式中的compute|0.475136|0.466432|+2.30%|[1.011601,1.032864]|

### Conversion-only

|范围|控制total ms|候选total ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|24样本四模式|0.047636|0.047555|+0.04%，无确认变化|[0.998504,1.001712]|

两边使用完全相同的转换与guard函数；测量差异不能声称为转换优化。

### Cold

|范围|控制total ms|候选total ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|24样本四模式|0.541440|0.531968|**+1.09%**|[1.001901,1.038536]|

### Steady-state

|范围|控制total ms|候选total ms|配对吞吐变化|speedup 95% CI|
|---|---:|---:|---:|---|
|24样本四模式|0.509440|0.499712|+1.72%，区间跨1，尚未确认|[0.991968,1.028630]|

### MSE与稳定性

|24样本输出指标|v61|v79|
|---|---:|---:|
|相对O0的MSE median|0.006653010287410|0.006653010287410|
|相对O0的MSE mean|0.007578847013303|0.007578847013303|
|相对v61的输出差MSE|0|0|

全部360条结果finite FP32、输出/payload/scale逐位一致；全部真实样本guard为0。
首层4样本MSE为median0.000278282719670、mean0.000283055340644；这是样本集合不同，不是精度提高。

|范围/模式|CV≥3%：控制/候选|分母|
|---|---|---:|
|四样本×三轮GEMM|6 / 5|12|
|24样本×三轮GEMM|33 / 34|72|
|四模式conversion total|0 / 0|24|
|四模式compute GEMM|12 / 12|24|
|四模式Cold total|8 / 8|24|
|四模式steady total|20 / 22|24|

四模式任一阶段CV失败：conversion 0/0、compute 12/12、Cold 23/23、steady 22/22。
所有记录保留，没有靠剔除慢记录生成收益；**未通过严格全阶段CV<3%门槛**。
共享未锁频环境不等于已定位每个离群原因，GPU快照不足以证明某次干扰来源。

## 正确性与安全范围

每次验证96项：两种M/N形状、六种输入模式、四种计时模式、两条实现；另8项非法输入拒绝。
覆盖全零、全E2M1编码、INT8极值交替、随机、整tile不安全回退、同一grid安全/不安全tile混合，
不同row/column/group scale、非默认stream、与独立FP64语义参考rtol/atol=1e-3、与v61逐位比较。
拒绝UE8M0 code0/255及不满足向量对齐的A/W输入。
形状为64×128×4096和128×256×4096，不将有限sanitizer检查宣称为4096³全覆盖。
memcheck/synccheck各0 errors；racecheck 0 errors/0 warnings。三者各完成上述96+8项检查。

## 复现与证据

源码提交：codegen `12059f6`，配对运行器 `6572b90`。
候选cubin SHA256：`67452e606635714a28910f6d2063b4b9fa6294b8bf63081aee755414232122695`。
控制cubin：`d390ad87e2bfb4f3326785482cfebd2c67f36884854c904e41356e30551b9e8c`。
正式扩展保持`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/probe_o3_eight_chain_codegen.py --output reports/o378_roof_v79_codegen
python scripts/benchmark_o3_eight_chain_probe.py \
  --output runs/o378_roof_v79_trace24 --samples 24 --rounds 3 --modes compute_only
python scripts/benchmark_o3_eight_chain_probe.py \
  --output runs/o378_roof_v79_four24 --samples 24 --rounds 1 \
  --modes conversion_only compute_only cold steady_state
```

命令要求新输出目录；现有结果不要覆盖。仅加载隔离cubin，不重新安装/切换正式扩展。
`reports/o378_roof_v79_codegen/`保存源码生成结果、PTX/SASS、资源、liveness及审计receipt；
`runs/o378_roof_v79_*`保存原始Event、配对summary、MSE、输入hash、环境、资源与验证记录。
完整原始归档`tmp/o378_v79_complete_evidence.tar.gz`在两端SHA256一致：
`28573a4299c46b14eaf36eb3c41669b4cb1d6b3b54ab89e0e5d6366ecd907a81`。
Git保留文本证据，不提交cubin/共享库；本地19项CPU/证据回归通过，包含从原始Event重算统计、
配对完整性、四模式guard成本、MSE、SASS累加输入、资源存活及安全日志复核。

当前最佳O3独立GEMM候选更新为v79；O7/O8仍为v78+v73。
必要MMA/fragment供数、168regs和3CTA未变，目标尚未达到。
不以这次小幅收益为由重复相邻调度扫描，也不把它与不同历史run收益相乘。

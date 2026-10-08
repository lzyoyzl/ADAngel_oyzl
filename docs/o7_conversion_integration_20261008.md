# O7 转换组合验收结果（v139）

本轮补齐此前的组合验证缺口，不是新的 GEMM 优化，也不把旧收益相乘作为新成绩。
当前 GEMM 最佳仍为 O3 v89、O7/O8 v78；正式默认和5090保持不变。

| 路径 | 权重转换 | 激活转换 | GEMM |
|---|---|---|---|
| 已测 v118 对照 | packed NVFP4→Q4 | 原 v73 E4M3→Q8 | 原 v78 |
| 本轮组合 | 同一 packed NVFP4→Q4 | 已测 v106 warp lookup E4M3→Q8 | **同一个 v78 CUfunction** |

v106/v118此前分别测过，本轮只验证它们组合后能否进一步改善转换和端到端。
复用v126冻结库的**control**分支；未通过编译门槛的v126 packed-MX8候选不执行，旧门槛不改写。
不新增CUDA编译、不重新安装依赖、不重编译正式扩展。

先校验旧库/源文件SHA、两条转换及guard完整SASS，再做32项合成四模式、边界拒绝、
非默认stream与逐位输出/metadata/norm检查；已测decoder本身未改动。
随后24样本×3轮四模式，warmup1000/repeats200/inner100，交错单stream，保留全部CV和原始Event。
直接比较v118，避免只与更慢的原始转换比较而宣称“又优化了最佳”。

```bash
python scripts/benchmark_o7_conversion_combo.py --validate-only --output runs/o378_v139_preflight
python scripts/benchmark_o7_conversion_combo.py --output runs/o378_v139_full24
python scripts/analyze_o7_conversion_combo.py --input runs/o378_v139_full24 --output reports/o378_v139_analysis.json
```

## GEMM 查重与范围

已复查v89/v78源代码、当前预热NCU和v92–138台账。当前整数热循环不是旧FP32 scale链。
更多链/warp、固定stage展开、fragment预排/预取、系数表及high/low分路已有编译或完整配对负结果。
本轮没有找到同时保持两路原生INT4、原scale且有新证据支持的GEMM改动，因此不重复这些试验。
约0.220347ms仍是1410MHz理想重叠的容量必要下界，不是本轮已经达到的延迟。

## 完整实测结果

已完成24样本×3轮、四种模式，共576条记录、345600个原始Event值。
原始FP16输入及O7源量化逐字段与既有v99一致。运行commit为`f3cd64b86651e652b8492dd2dee3628441d5bf7a`。

| 指标 | v118 W + 原A，ms | v118 W + lookup A，ms | 配对吞吐变化 | 配对延迟下降 |
|---|---:|---:|---:|---:|
| Conversion-only：W | 0.018186 | 0.018176 | 0.00%，实现相同 | 0.00% |
| Conversion-only：A | 0.045048 | 0.044339 | +1.58% | 1.56% |
| Conversion-only total | 0.063201 | 0.062459 | **+1.09%** | 1.08% |
| Compute-only GEMM | 0.481280 | 0.481280 | **0.00%** | 0.00% |
| Cold total | 0.553984 | 0.552960 | +0.18%，尚未确认 | 0.18% |
| Steady-state total | 0.528384 | 0.527360 | **+0.19%** | 0.19% |

延迟先取各样本三轮中位数，再取24样本中位数；百分比来自同轮同样本的配对速度比，
不是直接除表中两列median，也不与其他轮次绝对延迟拼接计算累计收益。
转换total的速度比bootstrap 95%区间为`[1.009982,1.013115]`；
steady为`[1.001942,1.003846]`，仅支持本轮微小正收益；
Cold为`[1.000000,1.001852]`，触及1，不宣称已确认Cold加速。

两侧GEMM均为同一个v78函数，CTA64×128×128、2stage、168regs、3CTA/SM。
steady略有改善来源于在线激活转换减少，不是Tensor Core吞吐提高。

| 输出精度 | 两种组合 |
|---|---:|
| MSE vs O5：median | 0.005536172273439442 |
| MSE vs O5：mean | 0.005053635851002639 |
| 新旧输出间MSE | 0，全部逐位一致 |

payload、scale、factor/anchor、精确平方和和guard均与参考一致；24样本全部走既有安全整数路径。
32项合成四模式、9项guard边界、131072个已保留NVFP4 decoder组合通过。
定向row-conversion memcheck/synccheck均0错误；这是小M/N、K4096检查，不是全GEMM sanitizer。
首次sanitizer过滤参数错误在执行前被CLI拒绝，修正过滤参数后通过；失败日志也保留。

## 波动和保留决定

每侧每模式72条记录。CV≥3%的数量：转换total为0/0，compute为1/1，Cold total为2/0，steady total为1/1。
但是Cold内另行批量测量的W分项仍为53/58，不能称所有阶段严格稳定性通过。
两侧使用完全相同的W kernel，且conversion-only W双方全部CV<3%；问题不能简单归因于组合算法。
抽样SM clock为1245–1410MHz，与负载状态变化相容，但没有逐Event时钟，不能确定具体原因。
保留全部原始序列；不删离群值，不用conversion分项之和代替直接端到端时间。

**保留联合路径作为当前O7独立转换组合**：相对于已优化v118，确认转换总量与steady的小幅改善；
不升级为正式默认，不称Cold已确认加速，更不算新的GEMM优化。
既有v106已测过lookup机制，本轮只是补齐组合验收；不追加邻近查表/packing变体，避免重复优化。
O3/O7/O8的GEMM最佳仍为v89/v78/v78，距离理想容量下界的主要问题尚未解决。

[完整原始数据、SHA和CPU复算](evidence/a100_o378_roof_v139/README.md)。

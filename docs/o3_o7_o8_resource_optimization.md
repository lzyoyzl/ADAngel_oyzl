# O3/O7/O8 资源上界优化进展

更新：2026-09-30。目标：在保持量化、G128 scale 和累加语义的前提下，缩小 A100 实现与资源模型上界的差距。**目前仍在候选迭代，未达到约 0.282–0.286 ms 的模型下界，正式默认尚未切换。**

## 当前最强的配对证据

候选6保持 CTA `64×128×256`、两级 cp.async、两路原生 INT4 和 FP32 输出不变，将 N fragment 扩宽至64，交错四个独立 N atom 的 MMA。每个输出仍按原 G128 顺序重构、转 FP32、缩放和 FMA。

24 个真实样本，每样本 3 轮，warmup=50、repeats=200；同输入对照原正式 symbol。以下是准备好输入的 compute-only，不包含转换。

本表已采用修复后的循环轮换：每个后端36组旧实现先运行、36组候选先运行。早期“轮换后反转”抵消造成的偏序记录保留在v1/v2，不用于本表。证据：[v3记录](evidence/a100_o378_roof_v3/README.md)，二进制SHA-256前缀 `6c6a1d8c`。

| 后端 | 原实现 median ms | 交错候选 median ms | 配对加速比 median | 配对加速比 95% CI | 候选相对原实现 MSE |
|---|---:|---:|---:|---:|---:|
| O3 | 0.558592 | 0.532480 | 1.0497× | [1.0474, 1.0532] | 0 |
| O7 | 0.600064 | 0.557056 | 1.0745× | [1.0714, 1.0784] | 0 |
| O8 | 0.600064 | 0.559104 | 1.0726× | [1.0709, 1.0756] | 0 |

加速比先按同样本、同轮配对再汇总，因此不必等于两列总体 median 相除。所有候选输出与原实现逐位相同；相对参考的 MSE 也完全不变：

| 后端 | 相对 O0 的 median MSE | 组内 FP16 参考 | 相对组内参考的 median MSE |
|---|---:|---|---:|
| O3 | 0.0066530103 | O0 | 0.0066530103 |
| O7 | 0.0119003180 | O5 | 0.0055361724 |
| O8 | 0.0094403548 | O6 | 0.0044110849 |

ISA 审计确认同一候选函数包含 LDGSTS、`IMMA.16864.U4.S4` 和 `IMMA.16864.S4.S4`，无 INT8 替代。仍有少量 local load/store；审计按允许 spill 的策略通过，而非零 spill。

## 尚不能宣布正式完成的原因

配对加速约 1.05–1.075×，离目标仍远。72 条记录中，原实现/候选的 CV≥3% 数量分别为 O3 `49/52`、O7 `34/47`、O8 `35/46`。全部保留，bootstrap为同trace相关样本的描述性统计；不能宣称全阶段CV验收通过，快照也不足以确定每个离群值的原因。

后续平衡顺序的24样本四模式复测已完成：576条全部逐位一致，每个variant/mode有12AB、12BA。compute-only配对加速为O3 `1.0432×`、O7 `1.0695×`、O8 `1.0635×`；cold为`1.0407×/1.0701×/1.0602×`，steady-state为`1.0451×/1.0637×/1.0625×`。转换代码未变，转换耗时无实质收益。仍有多条CV≥3%记录，详见[完整四模式证据](evidence/a100_o378_roof_v5_four24/README.md)。

## 第二轮诊断与后续候选

更宽 N fragment（tune6）通过192项逐位检查、原生 INT4 审计及 memcheck/synccheck（0错误）。24样本四模式共576条也全部逐位正确；由于使用修复前顺序，性能仅作诊断，不能据此切换默认。

NCU 中，O3 发射活跃比例由43.95%变为45.60%，O7/O8由约44.7%变为47.2%；就绪warp有所增加，但shared服务工作量没有明显下降，收益仍有限。详见[第二轮证据](evidence/a100_o378_roof_v2/README.md)。

针对驻留量增加的候选8/9/10现已通过240项逐位检查和原生INT4审计，但合成初筛均更慢。下表为同二进制、每候选10轮的compute-only median，单位ms，不替代真实trace验收。

| 后端 | 原实现 | 候选6 | 候选8：512线程 | 候选9：N64 | 候选10：N64/K128 |
|---|---:|---:|---:|---:|---:|
| O3 | 0.561152 | 0.537600 | 0.745984 | 0.601856 | 0.646144 |
| O7 | 0.603392 | 0.560128 | 0.747520 | 0.644608 | 0.657408 |
| O8 | 0.604160 | 0.565248 | 0.750080 | 0.646144 | 0.659456 |

候选10零spill仍更慢，不能只追求occupancy或零spill。候选6另通过K768三后端racecheck（0 hazards/errors/warnings），但这是有限范围的安全检查。

候选11利用O7激活scale为2的整数次幂：在A/W及乘积均为正normal的guard下，用精确指数位运算替代逐组scale乘法；I2F/FMA顺序不变，不用magic-bias。216项合成逐位检查及24样本对照通过，但O7候选6/11为0.552960/0.565248ms，候选11相对6配对加速0.98016×，不采用。NCU显示FMUL虽减少，总动态指令反而增加6.39%。[完整证据](evidence/a100_o378_roof_v4/README.md)

候选12把指数偏置减法移到每CTA、每row/group的shared scale预取阶段。288项逐位检查、原生INT4审计、memcheck/synccheck（各72项，0错误）通过。24样本O7候选6/12为0.553984/0.557056ms，MSE仍相同；目前没有明显增益，暂不采用。后续重点不再反复替换这一条乘法。

候选13只针对O3的UE8M0读取：将W scale由自然[N,G]重排为[G,N]，使同group的相邻列连续。重排单独kernel计入W转换（新增logical bytes=2×N×G），cold计入，compute-only/steady-state缓存。两路INT4、G128及scale数值不变；O7/O8不接受该候选。

已通过72项合成逐位验证、原生INT4审计、memcheck及有限K768 racecheck。24样本3轮compute对照：旧实现/候选6/13分别为`0.558080/0.531968/0.527616ms`；候选13比6的配对加速仅`1.00724×`，CI`[1.00192,1.01167]`，MSE不变。NCU额外global sectors由8,126,464降为0，long-scoreboard下降，但shared fragment及IMMA/I2F/FFMA工作量没有减少。四模式目前仅完成单样本烟测，重排增加转换时间，尚不切换默认。详见[候选13证据](evidence/a100_o378_roof_v6/README.md)。

候选14针对O7/O8：既有group-major FP32 scale panel改用16字节cp.async直接写入shared，与A/W payload共用commit/wait和CTA barrier，避免同步global load→寄存器→shared store。保持原FMUL、I2F、G128 FMA顺序、tile和双缓冲，不再使用无收益的power2替换。

已通过144项合成逐位检查、38个实例ISA审计、memcheck/synccheck（各48项）、有限K768 racecheck。24样本3轮共432条compute记录全部逐位相同：O7旧实现/候选6/14为`0.596992/0.556544/0.553984ms`，O8为`0.600064/0.560128/0.557056ms`。新增收益仍小，四模式只完成单样本烟测。NCU显示long-scoreboard下降，但共享供数及后处理仍限制性能；不宣称达到上界。

候选15将每个K256 stage的两个scale panel展平分配给完整warp拷贝，只合并搬运、不合并G128数学处理。192项逐位检查、39实例ISA审计、memcheck/synccheck及有限racecheck通过，但八轮合成初筛O7/O8约`0.564736/0.563200ms`，慢于候选14的`0.555520/0.558080ms`，不采用。NCU总指令及额外copy事务下降，但local理论sector增加20%，eligible warp下降；更规整的搬运不等于更快。四模式单样本32条均正确。[完整证据](evidence/a100_o378_roof_v8/README.md)

候选16/17转向延迟隐藏：保持输出CTA64×128，采用K128三阶段cp.async和三CTA launch-bound预算，分别使用N32/N64寄存器片。共享内存约50/51KiB，预取两个G128、等待最老stage后计算；排空时wait_group0。每个G128仍独立scale及有序FP32 FMA，没有重关联。CPU槽位复用/排空检查、GPU编译和45实例原生INT4/流水线ISA审计通过，但v9发现公共重构改变了12个旧基线的SASS，性能测试被拦截。现恢复直接prefetch调用结构，等待重编译后与v8逐指令比较；不能将名义三CTA视为性能收益，也不能宣称候选已通过正确性验收。[v9回归证据](evidence/a100_o378_roof_v9/README.md)

代码先本地提交推送，再同步A100；直接HTTPS fetch断流时，使用校验Git bundle导入同一已推送提交。默认仍是旧实现，不修改5090。改动工作量后重算资源上界，不把旧约0.28ms下界当性能承诺。

v10恢复直接prefetch调用后仍有12个旧基线SASS变化（例如O7/O8正式函数静态指令仍由832增加到19768），排除了“只需移除该lambda即可恢复”的假设；再次拦截性能测试。下一轮将两阶段源码完整恢复为已审计v8版本，并把三阶段模板隔离到`o3_pipeline_candidate.cuh`，只允许16/17调用。新增源码hash门禁，但最终仍须确认实际SASS，不把源码一致当作二进制一致。

v11证明仅隔离头文件仍不足：两阶段源码hash与v8一致，12个旧基线SASS仍变化，O7/O8函数仍为19768条静态指令。没有启动性能测试。下一步将三阶段device模板移至单独`roof_pipeline.cu`编译，通过host-only函数指针接口选择16/17，原`o1_o3.cu`不再实例化新device模板。该分离仅用于SM80，SM120构建源列表保持不变；仍等待实际编译/二进制验证，尚不能断言编译器内部根因。

v12独立编译已恢复12个旧正式实例和39个旧候选的SASS逐指令一致。45实例ISA、396项合成逐位检查、memcheck/synccheck（各252项）及有限racecheck通过；单真实样本四模式48条全部逐位相同。但16/17三CTA候选没有性能收益：O3约0.859/0.969ms，O7约0.596/0.598ms，O8约0.596/0.602ms，均慢于同轮候选6。NCU显示O3/17理论local sectors增加约12.9倍，两种后端均新增8,388,608个shared excessive wavefront，定位在LDGSTS拷贝。O7/17的local流量下降但总指令和片上服务需求增加，不能一概归因为spill。候选不采用；下一轮18/19仅放宽三CTA寄存器预算至两CTA，隔离这一影响。详细MSE、时序、计数及适用范围见[v12证据](evidence/a100_o378_roof_v12/README.md)。

## 复现入口

重新编译 SM80 扩展后，使用新的输出目录：

```bash
python scripts/benchmark_a100_roof_candidates.py --synthetic --validate \
  --output runs/roof_screen_new --tunes -1 6 11 --rounds 6
python scripts/audit_a100_o1.py --variant roof_candidate --allow-spills \
  --output reports/roof_audit_new
python scripts/benchmark_a100_roof_trace.py --tunes -1 6 11 \
  --output runs/roof_trace_new
# 四模式版本必须使用包含 roof_tune 参数的新编译产物；这是后续验收命令。
python scripts/benchmark_a100_roof_trace.py --all-modes --tunes -1 6 11 \
  --output runs/roof_four_modes_new
```

不需要重新采集 trace，不修改 5090 后端。记录、原始计时、来源 hash 和审计：[第一轮证据](evidence/a100_o378_roof_v1/README.md)。

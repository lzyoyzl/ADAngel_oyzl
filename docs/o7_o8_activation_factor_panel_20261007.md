# O7/O8：全K激活factor面板（v127）

状态：**完整24样本三轮配对已完成，O7/O8吞吐分别下降1.71%/1.72%，不采用。** 输出/MSE不变；当前最佳、正式默认或5090均不改。

## 瓶颈与新机制

当前v78已采用全K整数累加，不再逐G128做FP32转换。仍有两路INT4、整数factor乘加，以及metadata/payload的供数和同步。已有O8预热NCU见[瓶颈报告](o3_o7_o8_bottleneck_portability_20261007.md)，不能将PC采样占比当耗时比例。

新候选把CTA需要的32×64个**INT32激活factor**一次性加载进shared，始终按原G128索引读取；不把不同group的scale合并，也不省略加权。原每组Af cp.async/地址构造移到prologue。W factor、低/高INT4 payload仍用原两stage流水线，源码八链、64×128×128 CTA、4 warp、guard/fallback和数学输出完全不变；机器码调度变化见下文。

每线程4次16B copy覆盖32组×64行，加入第一批payload的commit；首次wait0和CTA barrier后读取，之后只读，不新增循环barrier。shared从34304增至41984B；需要查询实际驻留，不凭估计宣称occupancy不变。新增启动等待也计入GEMM，不能当免费预处理。

查重：不是v43寄存器scale提前加载、v81每stage metadata warp映射、v95 barrier替换、v105/v108 factor同值分组，亦不是v121循环内global只读访问。曾考虑全部A/W factor的16位压缩，但其解包/额外shared成本没有明确优势，因此**未实现、未测试**；本轮仅保留INT32 Af完整panel，没有压缩误差或新范围条件。

## 预设投入门槛

- 原v78控制完整编码不变；同一候选entry仍为两路原生INT4，非INT8替代。
- 热循环MMA 32+32、LDSM16、CTA barrier1不变；cp.async由10条变9条。
- 寄存器≤168、无热local、CUDA实际驻留保持3CTA/SM。
- 热循环静态指令至少减少3%；静态计数仅为投入筛查，不是速度预测。

通过才验证GPU坐标/同步、guard和逐位输出，再做24真实样本三轮1000/200/inner100配对；不做小规模性能筛选。正向后补四模式和MSE，负向或门槛失败则停止，不扫描panel窗口/压缩位数/邻近参数。

## 平台迁移边界

“低容量metadata可与大payload采用不同供数粒度”是可复用思路，但全K panel需权衡shared预算、启动开销和驻留数。A100用cp.async；其他平台必须重做copy、屏障、布局和资源审计，不能直接复制参数或声称已有跨平台收益。

## 编译后独立资源复核（候选GPU执行之前）

编译commit `f8ab39d61c1a92d7c26d0e1d7e65dc7d2098f99d`。旧控制完整编码不变；热循环383→356条（−7.05%），allocated GPR168不变，实际驻留仍3 CTA/SM。shared34304→41984B；MMA32+32、LDSM16、barrier1不变，async copy10→9。

原“零热spill”门槛**保留失败**：新循环有1条32位LDL、0条STL，整个function local size为8B/线程。SASS `0xd10 STL[R1],R13` 在循环前保存，`0x10f0 LDL R14,[R1]` 每组读回，`0x1210` 与shared base相加后用于 `0x1240 LDSM[R97]`，是地址分量而非partial。并非没有代价：静态展开的未完成MMA链峰值8→6（不是硬件同时在飞计数），且prologue读取需要时间。

按用户此前允许少量spill的授权，**在任何候选GPU计时前**单独允许这一次地址reload，其他原门槛及3CTA保持要求不变；保存初始失败和独立复核，不将失败回写成通过。运行入口`benchmark_o78_activation_panel.py`会在启动候选前写出review receipt，先做不同row/group factor及非默认stream数值/同步检查，再直接24样本三轮1000/200配对。没有小规模性能筛选，也不扫描邻近panel大小/寄存器参数；若无收益即停止。

## A100完整结果

本地实现先推GitHub，再在A100项目内fetch/ff-only。运行commit `de07e9c28aba24809a85eb0a1050fbf15db78d02`；旧控制为v78，双方共用原v73准备。原始FP16 trace分别直接量化为源格式，不重新采集，不更改公共量化口径。

24真实样本×3轮，同进程、输入、stream，交错顺序；warmup1000、repeats200、inner100。共288条compute-only记录、57,600次正式Event执行；GEMM和total字段是同一次计时，不重复计数。无锁频、不等待GPU空闲、不删除离群记录。

| Case | 旧v78 GEMM median ms | 新v127 GEMM median ms | 配对吞吐变化 | 配对吞吐比95%描述性区间 | 旧/新CV≥3%记录 |
|---|---:|---:|---:|---|---|
| O7 | 0.474112 | 0.482304 | −1.7076% | [0.982508, 0.983051] | 4/72；2/72 |
| O8 | 0.475136 | 0.483328 | −1.7186% | [0.980932, 0.983157] | 0/72；3/72 |

配对吞吐比为每样本三轮`旧延迟/新延迟`比值的中位数，再跨24样本取中位数，不是两列总体中位数直接相除。bootstrap区间仅描述本次固定trace配对，不是跨机器/模型保证。

| 输出误差 | 旧/新MSE median | 旧/新MSE mean |
|---|---:|---:|
| O7 vs O5 | 0.005536172273439442 | 0.005053635851002639 |
| O8 vs O6 | 0.004411084910985704 | 0.004381379299073540 |

288条输出均与v67参考及旧v78逐位一致，新旧输出MSE=0；metadata、guard和准备payload检查通过。64项合成、12项边界、2项跨row/group factor检查通过。候选entry的有限small-M/N、完整K4096 memcheck/synccheck均0 errors，不是4096³或racecheck全覆盖。

首次启动在GPU执行前因JSON将直方图整数键转为字符串而误报audit replay drift；已加往返回归修正，**未改CUDA、原编译gate或性能数据**。失败日志保留。原始编译、资源复核、Event、MSE和安全证据见[v127证据](evidence/a100_o378_roof_v127/README.md)。

## 为什么不采用，以及剩余瓶颈

| 同一G128热循环/资源 | 旧v78 | 新v127 |
|---|---:|---:|
| 静态指令 | 383 | 356 |
| 原生S4/U4 MMA；LDSM | 32+32；16 | 相同 |
| 普通IMAD | 128 | 131 |
| 静态未完成MMA链峰值 | 8 | 6 |
| 热local load/store | 0/0 | 1/0 |
| allocated GPR；CTA/SM | 168；3 | 相同 |
| shared bytes/CTA | 34304 | 41984 |

这版只是把Af的读取提前，**每CTA仍读取相同的8192B Af，逐G128加权并未减少**。27条静态指令的节省包括编译器对uniform/地址/控制的重新安排，不能全部归因于少一条copy，更不能直接当动态指令减少7.05%。与此同时，地址reload增加了一条供数依赖，静态MMA调度重叠减少，启动预加载仍有成本，驻留数没有增加。

实测否定了本候选的总体收益；这些SASS差异与既有NCU共同解释可能的取舍，**本轮没有新NCU，不能把1.7%退化完全归因于某一项**。当前主要矛盾仍是MMA/整数后处理依赖、实际就绪warp不足和供数同步的组合，而不是只缺HBM带宽，也不是尚未移除的逐组FP32 FMA。

停止此panel路线，不扫压缩位数/面板窗口/寄存器限额，不迁移O3。没有追加负向候选的转换、Cold、steady计时，不能从GEMM差值冒充端到端结果。此前已确认的转换优化保留，当前最佳仍为O3 v89、O7/O8 v78；主GEMM目标尚未达到。

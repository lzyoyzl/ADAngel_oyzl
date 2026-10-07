# O3 / O7 / O8：当前瓶颈、针对性优化与平台迁移

日期：2026-10-07。对象为 A100 上的**最佳独立候选**，不是替换正式默认。
既有 5090 代码、trace、量化语义和计时方式不改。

**v130补充：** 针对v129的54条PRMT做了[一次直接U32构造修正](o7_o8_tensor_factor_words_20261007.md)，没有改变layout或放宽门槛。
PRMT全部消除，循环461→406，但仍高于原383、热local读/写仍3/3；因此关闭此系数外积路线，无新GPU性能/MSE结论。
它进一步区分了两件事：byte packing的具体冗余可以消除，但把scale外积转给Tensor Core的供数/额外MMA/寄存器成本仍在。

**v129最新进展：** [Tensor Core 生成 scale 外积](o7_o8_tensor_factor_20261007.md)已完成 A100 编译审计。
主点积仍为两路原生 INT4；系数外积另用16条U8 MMA。IMAD族158→115，但新增54条PRMT与热local读/写3/3，静态循环383→461（+20.37%），未过预设工作量/local门槛。
因此停止，不运行候选GPU；没有新的性能或MSE结论。该结果说明：**把标量工作转给Tensor Core，需要先覆盖输入重排、额外计算与寄存器成本，不能只看被消除的乘法数。** 当前最佳保持。

**v128最新进展：** [跨K的A加载与旧partial收尾交错](o3_crossk_load_overlap_20261007.md)已实现并在A100编译审计。
SASS确认8次下一组A加载后有32次旧partial加权，但静态循环323→374（+15.79%），allocated168不变；超过预设最多+5%工作门槛，停止，不运行候选GPU。
因此没有新的性能/MSE/NCU结论；不能把新增静态指令换算为实测退化。这再次限定后续投入：隐藏依赖不能以明显增加地址/布局与控制工作为代价，且活跃寄存器减少不等于实际occupancy提高。

**v127最新结果：** [全K激活factor shared面板](o7_o8_activation_factor_panel_20261007.md)已完成24样本三轮配对。
静态指令383→356、3CTA驻留不变，但新增一次热地址reload、静态MMA未完成链峰值8→6；O7/O8配对吞吐分别**下降1.71%/1.72%**，输出/MSE不变，不采用。
没有减少每CTA的Af总字节、原生MMA数或主要整数加权工作。该结果说明：**优化供数粒度要同时核对依赖、启动代价和实际驻留，不能用静态指令数单独预测性能。** 本轮没有新NCU或端到端成绩，旧最佳与已确认转换候选保留。

**v126补充：** [O7 packed E4M3转换](o7_packed_mxfp8_conversion_20261007.md)相比已测LUT的静态指令456→464、实际驻留不变，未达预设投入门槛，未运行候选GPU。当前最佳GEMM不变，也没有新增性能/MSE结果。它提供一个具体的移植边界：即便同一GPU，O8有效的packed转换方法也不能未经验证就宣称对O7有效；格式舍入和指数选择成本不同。

**此前 v125：最佳 GEMM 不变。** [v123 O8 转换融合](o3_o7_o8_current_best.md) 改善了转换与端到端延迟，但不是 GEMM 提升；v124 交换 MMA 操作数未通过编译投入门槛。[v125 高低路独立加权](o3_split_weighted_20261007.md) 完成全 24 样本三轮配对，吞吐下降 5.71%、输出/MSE 不变，已淘汰。其证据进一步说明：缩短 MMA 依赖链时，必须同时考虑独立输出数量、整数加权指令和寄存器成本，不能只看链长度。

## 1. 当前最佳及仍未消除的开销

以下来自已完成的 v104 同进程 24 样本配对实验；本轮不重测旧版本，也不把不同轮次延迟相除当作加速比。

| Case | 当前 GEMM 主基准 | Compute-only median ms | 输出 MSE median | 输出 MSE mean |
|---|---|---:|---:|---:|
| O3 | 全 K 整数累加、八条 merged MMA chain、M8 CTA 遍历（v89） | 0.433152 | 0.006653010287409885 | 0.007578847013302749 |
| O7 | 全 K 整数累加、八条 merged MMA chain；v73 转换（v78） | 0.463872 | 0.005536172273439442 | 0.005053635851002639 |
| O8 | 全 K 整数累加、八条 merged MMA chain；v73 转换（v78） | 0.467968 | 0.004411084910985704 | 0.004381379299073540 |

MSE 参考分别为 O0、O5、O6，因源格式不同，不用这三行直接评判各格式的精度优劣。
上述延迟及配对口径见 [v104 全 24 样本结果](o3_o7_o8_unmeasured_v98_runtime_20261007.md)，完整四模式历史结果见 [当前最佳记录](o3_o7_o8_current_best.md)。

目前已把低/高 INT4 点积合并到寄存器 partial，在 guard 保证系数、乘积及每一步前缀不溢出时，用 INT32 完成全 K 精确累加，最后才转 FP32 并恢复行/列基准 scale。不能再把当前瓶颈描述成“每 G128 都做 FP32 FMA 的长链”。

但每组独立 scale 并没有消失：O7/O8 仍需读取整数行/列 factor，生成系数，再把 partial 加权累加；两路原生 INT4 的数据搬运、MMA 依赖、high×16 重构以及 stage 同步也仍然存在。

## 2. 瓶颈证据与解释

最近一次预热后的 O8 NCU 是单个真实 `layer_12_o_proj`、4096³、无锁频/不清缓存的诊断，不是 24 样本 Event 成绩。其原始数据见 [v114 analysis.json](evidence/a100_o378_roof_v114/reports/o378_roof_v114_o8_warm_ncu/analysis.json)。

| 证据 | 当前观察 | 能得出的结论 |
|---|---:|---|
| Eligible warps / scheduler；issue active | 0.7464；46.90% | 许多时刻没有足够已就绪指令；SM 并非持续满发射。 |
| 未发射 PC 采样：wait / math / barrier / MIO | 35.54% / 29.11% / 11.74% / 9.29% | 优先研究 MMA/算术依赖与供数同步；这些是采样占比，不是耗时占比。 |
| Tensor / FMA pipeline 利用 | 58.95% / 37.96% | Tensor Core 与整数后处理共同占用执行资源；不能只按 INT4 peak 推断 kernel latency。 |
| 动态 IMAD / 总 warp 指令 | 42.164M / 105.521M | 整数 factor、重构、累加和地址处理占有实质指令成本。不能把所有 IMAD 都归为 scale。 |
| 原生 INT4 SASS；热循环 I2F / local | U4×S4、S4×S4；均为 0 | 当前路径确实是原生两路 INT4，逐组 I2F 与热 spill 已不是首要问题。 |
| 168 allocated registers；3 CTA / SM | 12 resident warps / SM | 寄存器资源约束可用于隐藏等待的 warp 数；盲目增加链或大 tile 可能反而降低 occupancy。 |
| L2 hit；DRAM 实际读 / 写 | 96.97%；19.87 / 68.10 MB | 本次采集不支持“主要缺 HBM 带宽”的判断，不能仅凭低 DRAM 利用否定供数等待。 |
| LDS / LDSM excessive wavefront | 0 / 0 | 本次 shared 读取没有该指标所示额外 wavefront，不能再将其笼统描述成严重 bank conflict。 |

O7/O8 共用这一 GEMM 指令结构，但不能据一个 O8 样本断言 O7 或所有样本的 stall 百分比完全相同。O3 仅有权重侧的逐组 factor，后处理负担不同，也不能直接套用 O8 百分比。

纯 MMA、issue、shared 等模型提供的是各自理想服务时间下界，不是能相加的 kernel 最快时间。依赖、资源占用、stage 等待和 tail 必须一起考虑；约 0.22 ms 的纯 MMA 模型不能写成当前 kernel 保证可达的延迟。

## 3. 已检查的供数方向：factor-only read-only（v121）

原路径：`Af/Wf → cp.async → shared factor panel → LDS → 整数系数`。
候选路径：`Af/Wf → __ldg 只读 global/cache → 整数系数`。

**只改 factor metadata**；A-low/A-high/W payload 仍用原来的 cp.async.cg、shared swizzle、LDSM 和两阶段 pipeline。保留 64×128×128 CTA、4 warp、34304 B shared reservation、八条 MMA chain、所有数学运算顺序和原 guard/fallback。保留 shared 大小是为了隔离供数改变，不混入 occupancy 变化。

它与 v43 的提前把 scale 缓存在寄存器中不同，也与 v112 的取消全部 payload shared 中转不同；不是重扫链数、CTA、stage 或 cache-policy 参数。已核对 v43/v76/v87/v106/v109/v112/v120。

预先固定投入门槛：旧控制完整编码不变；同一候选 entry 包含原生两路 INT4 和矩阵异步拷贝；每 G128 仍为 32 条 S4×S4、32 条 U4×S4、16 条 LDSM、1 个 CTA barrier；payload copy 为 8 条；allocated registers ≤168、热 local 为 0，并至少减少 3% 静态循环指令。通过后才能检查实际资源、只读/nonalias 契约和 GPU 数值，再直接做完整 24 样本三轮配对测试，不做小规模性能初筛。

`__ldg` 不是保证更快。NVIDIA 明确指出只读非一致缓存可能有更长延迟，是否受益取决于足够的并行度；此候选还可能增加 global 地址指令。Af/Wf 必须由先前同 stream 的转换产生，GEMM 内保持只读且不与输出别名；不能拿它读取同一 kernel 内正在被修改的 metadata。[PTX 8.7 只读 load 说明](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc)

实施源码为 [roof_o78_readonly_factor_probe.cu](../csrc/sm80/roof_o78_readonly_factor_probe.cu)，可重放的生成/编译 gate 为 [probe_o78_readonly_factor_codegen.py](../scripts/probe_o78_readonly_factor_codegen.py)。

### A100 实际编译结果：停止该方向

源码 commit `2a36f6aec73bf9329d86caec854d141615decb1e` 已先推送 GitHub，再在 A100 项目目录 fetch/ff-only 后编译；原控制完整机器码与 v78 一致。

| 编译指标（同一整数热循环） | 原 v78 | factor-only 只读候选 |
|---|---:|---:|
| 静态指令 / G128 | 383 | 534（+39.43%） |
| Allocated registers / thread | 168 | 168 |
| 循环活跃寄存器峰值 | 166 | 158 |
| 热 local load/store | 0 / 0 | 0 / 0 |
| Signed / unsigned 原生 INT4 MMA | 32 / 32 | 32 / 32 |
| LDSM / matrix+metadata async copy | 16 / 10 | 16 / 8 |
| CTA barrier | 1 | 1 |
| factor global read-only load | 0 | 20 条 `LDG.E.CONSTANT` |

原 shared 路径读取 20 个 factor 值只用 4 条 scalar LDS 和 8 条 LDS.64；另有三条 `LDS RZ,[RZ]`，不是有效 factor 读取。候选消除了 factor shared 读和两条 metadata copy，但用 20 条 global load 及更多 64-bit 地址构造、MOV、整数指令代替：例如 `IMAD.WIDE.U32` 0→20、`LEA.HI.X` 2→20、MOV 2→20。普通 IMAD 仍是 128 条，**并未消除逐组 scale 的核心加权工作**。

这是指令/寄存器结构证据，不是实测“慢 39.43%”。未通过预设至少 3% 指令减少的投入门槛，停止：不运行候选 GPU，不扫描相近 cache-policy/地址参数，不迁移 O3。没有新 Event、MSE、conversion、端到端或 GPU 安全结论；未测不能写成 0% 提升。

13 份原始文本和 SHA 已冻结在 [v121 编译证据](evidence/a100_o378_roof_v121/README.md)。当前最佳保留；正式扩展 SHA 仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
本地 25 项 CPU/source 契约/新旧证据回放测试通过；A100 首次 source 契约测试 8 项通过。这些都不是候选 GPU 正确性证明。

这轮的有效结论是：**小 metadata panel 的 shared 广播/向量读取已相当紧凑，绕过它并没有降低实际工作。后续应只投入能实质改变 MMA/整数后处理依赖或降低工作量的方案，而不是继续换 metadata cache 路径。**此前失败的 chain/tile/stage/warp、系数预计算与供数扫描不重复。

## 4. 能否迁移到其他平台？

**能迁移的是算法与优化原则，不是 A100 的机器码、最优参数或提升比例。**

| 策略 | 可复用的内容 | 换平台必须重做的内容 |
|---|---|---|
| 转换融合、packed 数据、静态权重缓存、预分配 | 减少中间读写及在线工作；公共数据与计时契约 | 源格式 decoder、packing 和目标加载布局；Cold/steady 仍分别验收 |
| 寄存器 partial、最终一次输出 store | 不把中间点积反复落盘 | MMA fragment/thread 坐标、寄存器分配、spill 与 occupancy |
| merged MMA chain、供数/计算重叠 | 在资源允许时隐藏依赖延迟 | atom shape、chain 数量、tile、stage、同步及 copy API |
| Tensor Core 系数外积（v129 编译门槛淘汰） | 可将行列系数乘法作为矩阵外积评估，尝试分摊执行单元 | 原生整数位宽、系数范围guard、fragment packing、额外MMA与spill；本候选没有已证实收益 |
| 跨K加载/收尾交错（v128编译门槛淘汰） | 旧fragment失效后复用寄存器；用独立收尾工作覆盖下一块加载 | 旧stage读者生命周期、barrier、地址/布局重构成本；不能只看加载提前就宣称更快 |
| guard 后的全 K 整数累加 | 对可精确整数对齐的 scale，减少逐组浮点处理 | 重新证明当前格式、K 和真实范围的系数/乘积/前缀边界；失败时保留正确 fallback |
| factor-only 只读缓存（v121 编译门槛淘汰） | metadata 与 payload 可按不同复用模式供数 | 只读/可见性契约、cache 层次、地址成本、延迟与命中率，不能默认获益 |
| 全K Af面板（v127 A100配对负向，不推荐直接移植） | 小metadata与大payload可采用不同搬运粒度这一设计思路 | startup、shared预算、local地址依赖、编译调度；少指令和相同CTA驻留均不足以保证更快 |

A100 的异步 global→shared copy 可在计算时重叠，并避免一般搬运经过额外寄存器；不同 GPU 的寄存器/shared 预算和支持的 MMA shape 不同，所以相同 tile 不等于相同 occupancy 或有效峰值。[NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.8.0/ampere-tuning-guide/index.html)

对 RTX 5090，寄存器 partial、供数重叠和减少 metadata 工作仍有价值，但不能照搬 A100 的“两路原生 INT4 peak/2”：本项目 CUDA 12.8 的 legacy U4/S4 路径在 SM120 实际审计为 INT8 IMMA 加位操作，见 [O1/O3 实现及性能报告](o1_o3_optimization_and_performance_report.md)。这是**当前受测路径**的编译结果，不是宣称所有隐藏硬件模式不存在。现有 5090 后端保持可运行，移植须另立候选并在同 entry 重新审计/测量。

对 AMD/其他平台，量化、scale 分解、guard 和测量方法可以沿用，但 CUDA/CuTe 指令、warp fragment 与 pipeline 要换成该平台后端；尚未做跨平台验证，因此不声称迁移后的性能已确认。

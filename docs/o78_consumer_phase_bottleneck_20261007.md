# 当前 O7/O8：把等待定位到 MMA 与 scale 消费阶段

## 结论与范围

**当前最佳 GEMM 不变，本轮没有新增加速或 MSE 结果。** 这是 v133 离线分析：复用 v114 的 O8 NCU 报告，逐 PC 对照 v78 SASS，未重新运行 GPU。
对象为 `layer_12_o_proj`、4096³、原 `adangel_roof_o78_eight_chain_candidate`。

当前瓶颈不是已经消除的“每组 INT32→FP32 + 浮点 FMA”，而是 **MMA 依赖、整数加权、供数与有限延迟隐藏共同作用**。
本轮将过去按 IMMA/IMAD 大类统计的等待，进一步分解到实际消费者阶段；不能把采样占比当成耗时或可获得的加速比。

## 1. 机器码实际在做什么

每个 G128 的寄存器 partial 按以下顺序形成：

```text
high INT4 的两个 K64 点积
  → partial × 16
  → low INT4 的两个 K64 点积累加
  → partial × A_factor
  → 再乘 W_factor，加到全 K 的 INT32 accumulator
```

源码中的 `partial * (A_factor * W_factor)` 被编译器重结合成了后两步。
当前 SASS 每个热循环有64条 `partial × A_factor` 和64条加权整数累加，没有独立的 `A_factor × W_factor` 预计算。
两步合计 **33.554432M 动态 warp 指令，占全入口105.521152M的31.80%**。这说明scale仍有实质指令成本，但不是说它占31.80%的时间。
原整数范围 guard、每组 scale、两路原生 INT4、最终 FP32 输出及正确回退均不变。

## 2. 等待具体落在哪里

全入口共有10557个未发射PC采样，其中 wait=3752、math-pipe throttle=3073、short scoreboard=624。

| 消费者阶段 | Wait 样本 | Math 样本 | Short scoreboard 样本 |
|---|---:|---:|---:|
| high MMA，第一段 K64 | 328 | 799 | 179 |
| high MMA，第二段 K64 | 740 | 820 | 326 |
| high partial ×16 | 335 | 194 | 0 |
| low MMA，第一段 K64 | 212 | 408 | 0 |
| low MMA，第二段 K64 | 628 | 587 | 0 |
| partial × A_factor | 394 | 72 | 21 |
| 再乘 W_factor 并累加 | 460 | 74 | 2 |

四类 MMA 合计占 **50.85%的 wait、85.06%的 math 样本**；两步scale合计占 **22.76%的 wait**。
high MMA 消费者占80.93%的short-scoreboard样本，提示还要审查片段就绪关系；不能据此把等待全部归因给某一条前置LDSM。
wait表示固定延迟依赖等待，math表示数学管线发射受限；采样PC是消费者位置，不是上一条生产者的因果计时。[NVIDIA NCU 指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons-not-issued)

另外，1239个barrier样本落在循环控制消费者PC，而不是 `BAR` 指令行。入口含 deferred barrier，不能误解为“循环分支本身很慢”。
条件退出 `CALL` 虽有262144条warp指令计数，但predicate为真的线程计数除32仅8192，正好为2048 CTA×4 warp：它不是每个warp每组都调用一次函数。

结合旧报告的 eligible warp/scheduler=0.7464、issue active=46.90%、168寄存器/线程、3 CTA/SM，现有证据支持优先处理**有成本约束的依赖隐藏**。
L2命中96.97%、LDS/LDSM excessive wavefront均为0；目前不支持优先重做HBM带宽或shared bank-conflict优化。
O7共用该指令结构，但上述百分比只属于这个O8样本；O3只有逐组W factor，不能直接套用。

## 3. 对下一步优化的约束

| 看似直接的方向 | 已有证据与决定 |
|---|---|
| 提前计算 A_factor×W_factor，缩短partial之后的链 | v72已测：强制coeff先算增加local，未获益；本轮不是新方案，不重做。 |
| 增加独立partial或拆高/低链 | v92/v125已显示寄存器、加权工作与复用代价；不能只按链长度判断。 |
| 缓存全部系数，去掉寄存器乘法 | v132移到权重转换后仍使循环383→564，IMAD族158→224；编译投入门槛失败，停止。 |
| 只修改源码加载顺序 | v131完整机器码相同；不重复计时。 |
| 改为shared交接或增加warp隐藏等待 | v122/v104完整配对均负向；不能单独追求更多resident warp。 |

**不再做这些机制的邻近参数扫描。** 下一项GEMM候选必须先证明：它减少尚未处理的真实工作/关键依赖，且不以更多片段读取、昂贵地址构造或明显资源退化交换收益；随后才投入24样本交错配对、MSE及安全审计。
本轮没有找到满足这一条件的新kernel，因此不编造候选或重复测试。约0.22ms的纯MMA理想容量下界仍不等于当前实现保证可达的延迟。

## 复核方式

[v133分析JSON与输入SHA](evidence/a100_o378_roof_v133/analysis.json)记录383条热循环指令的角色及所有类别计数；解析核对全入口1976条解码指令的操作数和predicate，并将动态计数与原NCU汇总闭合。
忽略NCU不显示的reuse提示、分支地址重定位与标点差异；这不是对二进制编码逐字节等同的证明。
3项CPU测试涵盖逐PC匹配、错误操作数/predicate拒绝、计数闭合和原始输入SHA回放。不替代GPU正确性、MSE或性能测试。

```bash
python -m pytest tests/unit/test_o78_mma_phase_stalls.py -q
# 输出目录必须尚不存在；仅读取旧报告，不运行GPU：
python scripts/analyze_o78_mma_phase_stalls.py --output reports/o378_phase_review
```

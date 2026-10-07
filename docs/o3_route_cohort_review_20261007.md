# O3 两路原生 INT4：全 K warp 分路候选结果（v137）

## 结论

**没有取得加速，保留当前最佳 O3 v89；O7/O8、正式默认和5090均不变。**
本轮只测试两路原生 INT4，没有增加单路 INT8 对照。
候选通过正确性、指令、同步和内存安全检查，但完整配对吞吐下降 **23.80%**，不采纳、不做相邻参数扫描。

## 测试结果

A100、4096³、原24份真实trace；同进程、相同输入/stream、交错顺序，3轮，
每条记录预热1000次、CUDA Event测量200次。不锁频、不等待空闲、不删除离群值。
这是完整24样本测试，不是小规模性能初筛。

| 指标 | 当前最佳 v89 | 全 K warp 分路 v137 |
|---|---:|---:|
| Compute-only GEMM，中位数 ms | 0.438272 | 0.573440 |
| 配对吞吐相对旧最佳 | 1.00000× | 0.76199× |
| 配对吞吐95% bootstrap区间 | — | [0.76122, 0.76302] |
| CV ≥3%的记录 | 0/72 | 0/72 |
| MSE vs O0，24样本median | 0.00665301028741 | 0.00665301028741 |
| MSE vs O0，24样本mean | 0.00757884701330 | 0.00757884701330 |
| MSE vs 旧最佳 | 0 | 0，输出逐位一致 |

每个样本先取3轮延迟中位数，再对24样本取中位数。速度比先逐样本逐轮配对，再汇总，
所以不等于上表两列整体中位数直接相除。配对延迟增加31.24%，不是“延迟增加23.80%”。
三轮配对吞吐变化分别为−23.84%、−23.80%、−23.80%，无需为这一明确负结果重测。

量化、转换kernel与缓存语义没有改变。四种模式均通过功能/计时契约检查；
本轮正式性能仅测compute-only，**没有新增conversion、cold、steady-state的24样本成绩**。
GEMM明显变慢后不再扩展端到端测试，避免消耗无收益的实验机会。

## 改了什么，为什么值得单独检验

之前预热NCU表明，当前最佳仍受MMA依赖、数学发射压力和不足的ready warp限制，
并非逐组FP32转换/FMA或HBM带宽优先。[前一轮诊断](o3_best_warm_profile_20261007.md)

原v89在同一warp内完成 high MMA → ×16 → low MMA → 乘W系数/全K整数累加。
本候选将4个warp用于低位U4×S4、另4个用于高位S4×S4，分别累加全部32个G128；
最后只做一次shared交接，得到 `low_sum + 16*high_sum`，再执行原最终scale和FP32输出。
它保留独立G128 scale，**不是跨组忽略scale，也不是改为INT8矩阵乘法**。

这与v125/v135在同一warp中逐组拆链/重构不同，也与v122逐组MMA/scale warp交接不同。
本轮同时改变了warp分工、N32片段流式宽度与最后合并，不是严格单因素微基准。

| 实现/资源 | 旧最佳 | 新候选 |
|---|---:|---:|
| CTA / pipeline stages | 64×128×128 / 3 | 相同 |
| 线程/CTA | 128 | 256 |
| 每线程分配寄存器 | 168 | 128 |
| 资源允许的最大CTA/SM | 3 | 2 |
| 对应最大warp/SM | 12 | 16 |
| 动态shared/CTA | 50,688 B | 相同，末尾复用存储 |
| 安全整数主循环的local load/store | 0/0 | 0/0 |
| 整个函数local分配/线程 | 16 B | 112 B |
| W fragment读取量（实现计数） | 1× | 2× |
| A/W fragment合计读取量（实现计数） | 1× | 1.5× |
| 最后shared交接读写字节/CTA | 无 | 65,536 B |

最大warp数由实际CUDA占用率API查询，**不是NCU测得的实际occupancy或eligible warp**。
新候选的FP32 fallback有明显spill；真实24样本全部走安全整数路径，其主循环没有spill。
不能把这次真实样本变慢归咎于未执行的fallback spill，也不能宣称整个kernel零local。

## 从负结果得到什么

缩短依赖链和增加可驻留warp并不自动提高吞吐。这里没有减少INT4矩阵计算总量，
但两组warp重复读取W和scale，并新增分路控制与最后shared交接；每SM可同时处理的输出CTA由3降到2。
因此即使寄存器/线程减少，供数、指令发射和输出tile级并发的取舍仍可能更差。

这是与实测一致的实现代价解释，**不是对各项延迟贡献的定量归因**。
本轮没有追加NCU，不能声称增加的W读取单独解释全部31.24%延迟。
270条候选静态循环与旧323条不能直接比较：候选有两个互斥分支、线程数也不同，不能据此声称工作量减少。

后续若继续，需优先证明减少真实供数/发射工作、保持跨输出复用，而非仅提高理论occupancy；
不再测试这个分路结构的邻近CTA、warp数或将其直接搬到O7/O8。
当前目标尚未达到，本轮不计入优化收益。

## 验证与复核

96项合成正确性/四模式/非默认stream检查、8项非法输入拒绝、2项分组CTA/尾组映射检查通过。
memcheck、synccheck、racecheck针对同一候选入口，覆盖128×384×4096安全和fallback混合网格，均无错误/警告。
这是小网格安全验证；24个4096³样本另做输出逐位和MSE回归，不将前者表述为完整4096³ sanitizer。

同一正式候选入口的PTX/SASS确认U4×S4、S4×S4原生INT4及cp.async，未出现INT8 MMA。
控制入口完整SASS保持不变。首次编译的同名函数ADL冲突已修复，失败日志一并保留；不是性能迭代。
本地重算统计允许1e−12的CPU浮点尾差，全部原始GPU计时不改动。

[原始数据、SHA与复算入口](evidence/a100_o378_roof_v137/README.md)

```bash
# 已有固定cubin/数据的A100项目目录中；output需为新目录。
python scripts/benchmark_o3_route_cohort.py --output runs/o3_route_validation_new --validate-only
python scripts/benchmark_o3_route_cohort.py --output runs/o3_route_full24_new \
  --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100 --modes compute_only
python -m pytest tests/unit/test_o3_route_cohort.py tests/unit/test_roof_v137_evidence.py -q
```

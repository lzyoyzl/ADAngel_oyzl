# v76：O7 CTA 内系数查表，负向结果，停止候选

**四个真实样本×三轮配对：GEMM 吞吐下降23.99%，没有刷新最佳。**
不扩大24样本，不切换正式默认，不修改O3/O8或5090后端。

## 唯一新思路

v67每个G128对输出partial执行整数scale乘法。O7的A factor为2的幂，
本轮尝试由每个CTA预生成 `table[d,col] = W_factor[col] << d`，
consumer按本行A factor的指数读取系数，再执行 `acc += partial * coefficient`。
这不是新量化，也没有删除任一G128的scale。

- 保持CTA `64×128×128`、4 warp、2-stage、N64片段流、两路原生INT4及最终一次写回。
- 两条测试路径均使用相同v73行级转换/metadata准备；新查表工作和入口检查全部计入GEMM。
- 表有10行。CTA入口检查全部A factor是否为正的2次幂且小于1024；不满足则使用原整数body。
- 原INT32系数/乘积/前缀安全界保持不变；原status=1使用逐组FP32回退，非法源仍在host拒绝。
- 每个producer线程生成一列；普通shared store由既有下一stage CTA barrier排序。
  没有新增每G128 barrier，但新增一次入口CTA归约/同步。
- 实际选中的系数由旧界保证不超过INT32_MAX；未被选中的表项使用定义明确的无符号移位。

首层四样本的checked metadata预测每张矩阵2048个CTA均走查表，没有回退。
这是从metadata精确计算的**host预测**，不是新增device执行计数器，也不外推为全部24样本覆盖。

## 指令与资源：收益被额外工作抵消

以下是同entry内一个G128自然循环的**静态**指令数，不是动态流量或耗时比例。

|项目|v67原整数主循环|v76查表主循环|
|---|---:|---:|
|总指令|378|381|
|IMAD类指令（含地址等用途）|154|112|
|IMMA|64|64|
|LDSM|16|16|
|标量LDS类读取|15|39|
|STS类写入|0|11|
|LDL local读取|0|4|
|寄存器/线程|168|168|
|Dynamic shared/CTA|34304 B|43520 B|
|Driver查询可驻留CTA/SM|3|3|

整个entry的ptxas报告：原实现8B stack、8B spill store/load；候选72B stack、
140B spill store及116B spill load。不能把整entry spill全归入查表热路径；
热路径明确新增的是4条LDL，原整数回退和FP32回退另有local访问。
同一候选PTX/SASS保留U4×S4和S4×S4原生INT4以及cp.async/LDGSTS，无INT8替代。

解释：虽然IMAD类工作减少27.27%，MMA和payload供数不变，查表增加shared读写、
地址计算与local读取，驻留规模也没提高。因此不能仅凭“少一次乘法”预测加速。
本轮没有新增NCU，不能进一步声称各项开销的精确贡献或bank conflict比例。

## 四样本配对结果

A100、原始FP16 trace直接生成现有O7源格式；公共源格式量化不计时。
样本为layer_00的q/k/v/o，4096³；warmup50、repeats200、inner100，单stream、预分配。
三轮交错顺序，24条记录、4800个独立GEMM Event区间全部保留。

|指标|同轮v67控制|v76查表候选|
|---|---:|---:|
|Compute-only median ms|0.453632|0.591872|
|配对吞吐变化|基准|**−23.99%**|
|配对speedup描述性95% CI|1.0000|[0.756567,0.798561]|
|CV≥3%的记录|12/12|2/12|
|相对O5的输出MSE median|0.000102067943158|0.000102067943158|
|相对O5的输出MSE mean|0.000099167492985|0.000099167492985|
|新旧输出差MSE|—|0，逐位一致|

延迟先对同一样本跨轮取median，再跨四样本取median；吞吐先配对再取median，
因此不等于两列总体median直接相除。四样本有关联，bootstrap仅为描述性对照。
全部输出finite FP32，metadata逐元素核对及相对旧59的MSE回归通过。
这些MSE是**首层四样本**结果，不能与24样本MSE median拼接或宣称精度提高。

共享、未锁频GPU；CV范围1.12%–5.51%，未达到严格稳定性验收。保留全部失败记录和GPU快照，
不将其直接归因于他人负载或频率变化。当前配对结果没有支持扩大的正向证据。
Conversion/Cold/steady未作性能测量，不能用阶段相加推算；四模式只进行了小规模正确性检查。

## 正确性、安全性和停止边界

64项GPU检查覆盖O7/O8、全零/随机/交替极值/宽scale、四模式和非默认stream，
与v67逐位一致、对FP64参考满足rtol/atol=1e-3；另12项源格式/范围边界检查通过。
O8这里只作为非2次幂factor及回退的合成验证，不是O8性能实验。
memcheck、synccheck均0 errors；racecheck为0 errors、0 warnings。
sanitizer范围M≤128、N≤256、K4096，**不声称覆盖4096³的sanitizer检查**。
四张4096³真实矩阵另外通过输出和MSE检查。

首次预检因非法源数据异常消息与复用验证器不一致而中止，尚未产生性能记录；
`abd6b59`修复该host接口约定并新增3项CPU回归，CUDA/cubin未修改。
后续使用全新`screen_r1`目录完成上述测试，没有删除慢记录或覆盖失败计时。
21项CPU契约/证据测试重新计算全部原始Event统计、配对结果、MSE、源码hash和指令审计。

**停止此查表候选，不追加表大小/布局扫描。** 当前最佳仍为v67 GEMM加v73在线准备。
本轮没有缩小到既有有效吞吐上界的差距，目标尚未达到。

## 可复核证据与复测

- CUDA/codegen提交：`96dfa85`；运行脚本：`dc58f7c`；接口修复：`abd6b59`。
- 编译原始归档SHA256：`d086545cffa3d4a774ccabdeae90f81fe09aad96253031d1647260b183a3e9d5`。
- 本轮运行归档SHA256：`2b8c4d4cfa46b883e0f996aa81de55a1081f5c3e39895a32c8b262a6f2e68dc8`。
- `reports/o378_roof_v76_codegen/`：完整PTX/SASS、generated header、codegen/source hash、
  build/resource/liveness及sanitizer日志；cubin留在完整本地/服务器归档，不作为Git依赖。
- `runs/o378_roof_v76_screen_r1/`：全部Event、summary、环境/source provenance、validation和GPU快照。
- 正式扩展SHA256仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

```bash
python scripts/benchmark_o7_factor_table.py \
  --output runs/o7_factor_table_recheck --samples 4 --rounds 3
python -m pytest tests/unit/test_roof_v76_evidence.py \
  tests/unit/test_o7_factor_table_candidate.py \
  tests/unit/test_o7_factor_table_analysis.py \
  tests/unit/test_o7_factor_table_runtime_contract.py -q
```

性能复测需A100上既有v67/v73/v76编译目录及trace；输出目录必须不存在。

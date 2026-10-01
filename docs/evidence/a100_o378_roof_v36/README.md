# v36：O3直接目标布局转换（主测与长预热诊断完成）

GEMM54、量化和G128 scale语义不变，仅内部`conversion_impl`选择：

|ID|路径|
|---|---|
|0|现有Q4转换、W scale转置、payload重排；激活split后重排|
|1|精确寄存器Q4 LUT、标量目标布局转换；同时转置W scale|
|2|同1，向量化输入输出；A读取8B、W读取4B，每线程处理8元素|

不改变正式默认、O7/O8或5090。每次转换完整包含查表、packing、payload和scale重排；
不把转换离线，计时区间无申请。向量路径明确验证指针对齐和索引范围。
诊断自然布局仅在计时后导出，GEMM直接使用最终G128-major数据。

## 24样本、50次预热的主测

同binary循环换序，warmup50/repeats200/转换inner100，共288条完整记录。
下列收益按同样本配对比值计算，不是两列总体median的商。

|模式|原路径0 ms|标量1 ms / 吞吐变化|向量2 ms / 吞吐变化|
|---|---:|---:|---:|
|Conversion-only|0.133555|0.077384 / +72.13%|**0.043615 / +206.06%（3.06×）**|
|Compute-only|0.492544|0.488448 / +0.52%|0.488448 / +0.83%|
|Cold|0.631296|0.591360 / +7.69%|**0.552960 / +15.57%**|
|Steady-state|0.558080|0.545792 / +2.93%，未确认|**0.526336 / +5.11%**|

向量2的配对speedup描述性95%区间依次为[2.99313,3.06901]、[0.99581,1.01499]、
[1.13704,1.17446]、[1.02637,1.07385]。标量1的steady区间[0.99634,1.03416]跨1。
转换、cold和向量steady在本轮有收益；GEMM符号及机器码完全相同，
compute差异不能解释为新的GEMM数学优化。纯GEMM主结果仍引用v30五轮，不从本轮挑最小值替换。

所有payload、scale及最终输出与旧转换+GEMM54逐位一致，优化引入的输出MSE=0。
相对O0的Median MSE=0.006653010287410，Mean MSE=0.007578847013303。
这不是直接相对FP16原始模型的精度，也不是相对O0的性能加速比。

## 稳定性限制与独立诊断

按conversion/compute/cold/steady，选定阶段CV≥3%的记录数（每项24）：

|路径|选定阶段CV失败|任一阶段CV失败|
|---|---|---|
|0|0 / 11 / 4 / 13|0 / 11 / 6 / 13|
|1|0 / 12 / 5 / 19|0 / 12 / 22 / 22|
|2|0 / 9 / 12 / 24|0 / 9 / 23 / 24|

向量steady的CV中位数4.13%，最小3.02%、最大7.00%，并非偶发的单点离群。
例如layer12 q的旧路径前20次约0.492ms、最后20次约0.558ms；向量路径约0.467→0.541ms。
因此不能声称全阶段CV<3%，也不能把漂移无证据地归因于某个外部进程。
补做500预热单样本诊断后，steady CV约0.59%/0.58%/0.63%，支持预热/状态尚未稳定的解释，
但未区分时钟、功耗、缓存及其他系统因素。完整24样本500预热诊断独立保存，不覆盖50预热主测。
共享GPU、未锁频、不删离群；24样本来自同一trace，bootstrap仅作描述性比较。
正式默认不切换。

### 完整24样本500预热诊断

保持同binary、同输入、同循环顺序、repeats200/inner100；只将所有路径warmup统一从50改为500。
另有288条记录，不能选择性替换原主测的坏记录，不能用两个run的延迟相除。

|模式|旧0 ms|标量1 ms / 配对吞吐变化|向量2 ms / 配对吞吐变化|
|---|---:|---:|---:|
|Conversion-only|0.133548|0.077389 / +71.87%|**0.043551 / +206.60%**|
|Compute-only|0.494592|0.492544 / +0.42%|0.492544 / +0.42%|
|Cold|0.634880|0.592896 / +7.23%|**0.551424 / +14.89%**|
|Steady-state|0.561664|0.545536 / +3.20%|**0.527360 / +6.56%**|

向量2的conversion/cold/steady speedup区间为[2.99359,3.07165]、[1.14525,1.15456]、[1.06263,1.07292]。
compute同一GEMM下约0.42%的微小差异不用于宣称数学kernel优化，仍保留全部原始数据。
这次重复支持转换及端到端收益，全部输出仍逐位相同，MSE完全不变。

按conversion/compute/cold/steady，选定阶段CV≥3%数量：0路径`0/1/2/0`，1路径`0/1/1/0`，2路径`0/0/0/0`。
但任一阶段CV失败分别为`0/1/2/0`、`0/1/23/0`、`0/0/24/0`。
也就是说向量候选的总延迟及compute已较稳定，**cold内部GEMM子阶段仍未达到CV门槛**；
不能将“total稳定”扩大为“全部阶段稳定”。更长预热不保证所有测量稳定，尚未定位剩余分段波动的具体原因。
主测/诊断见`runs/o378_roof_v36_four24/`和`runs/o378_roof_v36_four24_warmup500/`，
单样本500预热诊断也完整保留，未计入24样本统计。

## 实现和审计证据

- W的3个kernel、A的2个kernel分别合并为1个，仍计入全部转换成本。
- 4096³减少W中间payload读写16MiB、A中间payload读写32MiB；W scale必需的读写仍保留。
- 向量A实际SASS为LDG.E.64 + 两条STG.E；W payload为LDG.E/STG.E，scale仍是字节读写。
- 4个转换实例13–17寄存器，STACK/LOCAL=0，无LDL/STL、F2I或通用地址除法。
- 96项GPU检查、8项非法入口检查通过；独立FP64语义参考、全编码、饱和/零/随机/交替、非默认stream覆盖。
- 有限memcheck/synccheck各32项+8项拒绝检查，均0 errors；不是全输入空间安全证明。
- 148个候选的原生双INT4/cp.async审计通过；12个旧正式与148个旧候选GEMM的SASS不变。
- 两个非目标mixed-binary函数codegen变化，全范围比较passed=false原样保留；完整mixed回归通过，含480项binary GEMM。
- 新转换没有改善GEMM的MMA/I2F/FP32后处理工作量，也没有改变约0.220347ms的必要容量下界。

实现commit：`d0bc5a41a5673780ee8a2e97cd2ae6fa1aabd4cb`。
binary SHA-256：`04612afa2fecd53215260a1d3aea334279c1feef35d95dc44c00abc0d6d6ffab`。
原始每次Event、summary、环境、provenance、MSE及诊断见`runs/`；审计、回归与编译日志见`reports/`。
同binary性能测量与SASS是不同证据，不以静态opcode数代替动态流量或延迟。
50个文本证据经哈希及路径检查后导入。归档SHA-256：
`b39c137633c0684ac8ef2f109ce831a8ff1e6f91e2e47d4c200c970ab52e35e5`。
大型完整SASS/PTX留在A100项目内；新增4个转换核的完整SASS及静态访存opcode摘要已随文本证据保存。

```bash
python scripts/validate_o3_conversion_pipeline.py --output runs/o3_conversion_validation
python scripts/audit_o3_conversion_pipeline.py --output reports/o3_conversion_audit
python scripts/benchmark_o3_conversion_pipeline.py --output runs/o3_conversion_four24
# 独立预热诊断，不覆盖上一目录或改变主测协议。
python scripts/benchmark_o3_conversion_pipeline.py --warmup 500 --output runs/o3_conversion_four24_warmup500
```

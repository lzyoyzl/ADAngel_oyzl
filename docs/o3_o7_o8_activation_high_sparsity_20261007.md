# O3/O7/O8：激活高位稀疏化的检查（v113）

结论：**完整24样本不支持继续投入“稀疏高位＋标量精确补回”路径，停止。**
没有实现候选 GEMM，也没有新增延迟、MSE或端到端结果；当前最佳和正式默认不变。

## 先查重，再做数据检查

|已有检查|机制|本次区别|
|---|---|---|
|v103|Q4权重的稀疏主项＋精确残差|本次对象是激活高位 H，不是权重|
|v100|精确平衡 radix16，稠密双路 S4|只复用表示做数据统计，不重测该稠密 kernel|
|v112|global直接供给寄存器|本次无供数、寄存器预算或 tile 参数扫描|

检查24份现有真实 trace、5种固定策略，共120条记录。O7/O8的48份激活源格式
identity与已测v99逐项相同；O3沿用已有 `A_int8`。不改变量化、scale、K顺序或源数据。

## 检查什么，以及为何停止

将激活整数精确表示为 `q=L+16H`。普通表示为 U4低位＋S4高位；
O7/O8还检查平衡表示 `H=floor((q+8)/16)`、`L=q-16H`，两者都为S4。
O3仍遵守原 UINT4＋INT4 要求，不引入平衡表示。

SM80 INT4稀疏 A 的限制是**每8个元素的4个相邻对中保留2对**，不是任意2:4。
[NVIDIA PTX 12.8 稀疏存储规则](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#sparse-matrix-storage)。
每个固定K8保留非零数最多的两对，其他非零数完整保留为残差 D，不剪枝：

```text
H = H_main + D
P_G128 = L @ W^T + 16 × (H_main @ W^T + D @ W^T)
```

原组 scale 仍作用于完整 `P_G128`。CPU测试验证整数恒等式和分组语义，
但不能替代未实现 GPU kernel 的输出/MSE验证。

预先声明的投入预算为 `0.220347/4=0.05508675 ms`：只有高位一路稀疏化，
即使该路 MMA 工作减半，两路合计的理想容量收益也最多25%。
标量补回模型为 `N×残差非零数/(108×1410×1000×128) ms`；128是宽松的
每SM每周期lane补回容量假设，不是实测IMAD吞吐，尚未扣加载、解码、packing、scale或同步。

|策略（各24样本）|高位零值率 median|最小精确残差率 median|标量补回模型 median ms|单样本通过数|
|---|---:|---:|---:|---:|
|O3 普通|52.01%|13.34%|0.470252|0/24|
|O7 普通|30.91%|24.97%|0.880316|0/24|
|O7 平衡|36.17%|22.70%|0.800227|0/24|
|O8 普通|50.60%|13.67%|0.481915|0/24|
|O8 平衡|68.52%|7.34%|0.258838|4/24|

预设继续投入要求是同一策略24/24通过，五种策略均失败。
O8平衡表示的4个通过样本全部来自layer 0，不能推广为整个数据集的优势。

**这个门槛只是有限额度下的投资筛选，不是算法不可能加速的证明。**
MMA与SIMT工作可能重叠，不能将补回模型与MMA下界相加当作kernel最短延迟，
也不据此否定所有向量补回、重排或其他稀疏算法。不放宽门槛或继续相邻参数扫描。

## 保留当前最佳，不冒充新增性能

O3 v89、O7/O8 v78+v73主基准及已有v99微调不变。
可对照的最近一次同轮全24最佳侧 GEMM median为O3/O7/O8：
`0.433152 / 0.463872 / 0.467968 ms`；对应输出MSE median（/O0、/O5、/O6）：
`0.006653010287410 / 0.005536172273439 / 0.004411084910986`。
这些引用自[v104完整配对测试](o3_o7_o8_unmeasured_v98_runtime_20261007.md)，不是本次测得；
旧轮CV失败和未锁频说明仍适用，不能拼接成新成绩。目标尚未达到。

实现源 `fb82df383007856d3b2bf6c558923d4bbf672061`先推GitHub，再在A100核验bundle、fetch/ff-only。
本地20项CPU契约、冻结证据和v103回归测试通过；不算作GPU数值或性能验收。
正式扩展前后SHA相同，trace、production默认、转换kernel及5090均不改。

- [原始120条记录与summary](evidence/a100_o378_roof_v113/README.md)。
- [环境、源SHA与预设预算](evidence/a100_o378_roof_v113/reports/o378_roof_v113_activation_high_sparsity/environment.json)。
- [逐样本数据](evidence/a100_o378_roof_v113/reports/o378_roof_v113_activation_high_sparsity/results.jsonl)。

如需复现数据检查，用**新的**项目内输出目录，勿覆盖本次证据：

```bash
python scripts/inspect_activation_high_sparsity.py \
  --output reports/activation_high_reproduction
python -m pytest tests/unit/test_activation_high_sparsity.py \
  tests/unit/test_roof_v113_evidence.py tests/unit/test_sparse_q4_feasibility.py -q
```

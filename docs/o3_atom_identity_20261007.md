# O3：单 G128、原生 N8 子块的 identity-factor 机会检查（v116）

本轮只检查范围和指令预算，不是新 GEMM 性能测试。正式默认、O7/O8和5090不改。

## 为什么检查这一点

原整数路径每组计算 `acc += P * factor`。若一个原生 N8 MMA 覆盖的8列在这一组的
factor都为1，理论上可把原有 `16*high` 与 final acc 合并后作为 low MMA 的 C 操作数，
省掉该组末尾的乘加更新。仍需原来的两路 INT4；不能删除 group scale 或修改量化值。
这是单 G128 的条件分派，不要求32组的scale全部相同，也不排列输入或输出。

查重：v77只在人工全unit输入上隔离成本；v105检查O7/O8四行factor向量重复；
v108检查完整32组同质profile与排列。这里仅补此前未覆盖的**真实数据原生子块条件**，
不重跑这些旧kernel，也不把它命名为已完成的新优化。

先用实际O3 CuTe `partition_C` 进行host-only坐标核验，再复用原SHA固定的24份scale快照。
最乐观情况下，每个原循环最多可省64条末尾更新，分母为当前整数循环323条指令：

```text
乐观指令工作减少 = 原生N8×G128子块可用率 × 64 / 323
```

忽略所有检测、flag读取、分派、寄存器、指令重排和回退开销；不是延迟或加速预测。
预先沿用5%的投入门槛；即使上述免费实现模型都达不到，就不编写新GEMM。

## 复现

```bash
TMPDIR="$PWD/tmp" python scripts/inspect_o3_atom_identity.py \
  --output reports/o378_roof_v116_atom_identity
python -m pytest tests/unit/test_o3_atom_identity.py -q
```

仅编译/执行host坐标校验；不创建CUDA context、启动GPU kernel、下载模型或重新量化。
源码先本地提交并成功推送GitHub，再同步A100执行；输出保留24份结果和完整来源凭据。
O7/O8不能仅凭A factor为1推断A/W乘积为1，本轮不向这两个后端外推覆盖率。

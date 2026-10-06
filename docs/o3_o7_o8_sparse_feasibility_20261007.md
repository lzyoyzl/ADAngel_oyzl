# v103：无损结构化稀疏路线的数据可行性检查

状态：仅检查现有 24 样本权重；不是新 kernel，不计作性能提升，正式默认不变。

## 先查重，再检查一个新机制

已核对迭代简报、v89/v90、v91–v102 和当前源码：tile/stage、producer、cache、
partial 链数/生命周期、整数全 K 累加、输出 streaming 等都有记录，不重做。
此次唯一新机制是能否将当前 Q4 权重**无损拆成稀疏主项与精确残差**，减少主项 MMA 工作。
源码检索未发现已实现的 native sparse INT4 + exact residual 候选。
不是把稠密矩阵拆成两份稀疏矩阵再都算一次：后者抵消 2 倍容量收益，还有 metadata 开销。

## 硬件规则与数学语义

[CUDA 12.8 官方 PTX 文档](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#sparse-matrix-storage)
规定，整数 s4/u4 的稀疏 A 操作数是 **paired 4:8**：每 8 个相邻 K 元素保留 4 个，
以两个相邻元素为一对，从四对中选择两对。不能用普通逐元素 2:4 规则统计 INT4。
metadata 使用两个不同且升序的 pair index；合法编码为 4、8、9、12、13、14。

对每行当前 K 顺序的每个 8 元素块，保留非零个数最多的两对，剩余值原样进入残差：

    Wq = Wmain + Wresidual
    dot(Aq, Wq) = dot(Aq, Wmain) + dot(Aq, Wresidual)

没有删去任何非零权重，没有更换定点转换或 scale，没有改变 G128 边界。
选中的 pair 允许在本可行性模型里存入数值零，这是对潜力有利的假设。
报告的最小残差只是在**固定 K 顺序/当前 pair 位置**下的下界，不是所有重排方案的全局最优。
稀疏 ISA 的操作数 A 方向与当前权重 B 不同，若实现还需解决逻辑转置、fragment 和输出布局；
本轮没有实现这些变化，也没有验证新的 sparse SASS。

## 检查方法和停止原则

- O3 读取现有 prepared W_q4，并与当前 E2M1→Q4 映射逐字节核对。
- O7/O8 使用现有公开 source preparation 与 fixed reference，不更换量化器。
  全部 48 个 weight source 的完整 payload、scale、微指数等 SHA 必须与 v99 记录完全相同。
- 24 样本 × O3/O7/O8，完整 4096² 权重；不做小矩阵性能初筛。
- 统计 Q4 零值比例、active pair 数、最少精确残差比例、需补回的 8 元素块比例、
  不需残差的 G128 比例。逐块验证主项加残差还原原整数，合法 metadata。
- 对“主项 sparse MMA + **标量 SIMT** 残差”仅给出有利的工作量模型：
  108 SM、1410 MHz，假设每 SM 每周期 128 个 lane correction，且每个残差 MAC 只需一条 IMAD。
  不计读取、gather、地址、metadata、scale、同步；不是实测 IMAD 吞吐或完整 kernel peak。
- 另给出理想四 MAC vector instruction 的容量值，但它**不含 gather/packing**，
  不能据此声称 DP4A 候选能达到该时间。MMA 和标量可重叠，因此不把两个下界直接相加。

若大量残差仍需逐元素补回，不开发标量残差 kernel；这只是淘汰当前机制，
不是证明所有稀疏/重排/向量方案都不可能。不得剪枝后冒充原精度实验。

## 运行与结果范围

先本地提交并成功推送，再同步 A100，执行：

    python scripts/analyze_sparse_q4_feasibility.py \
      --output reports/o378_roof_v103_sparse_feasibility
    python -m pytest tests/unit/test_sparse_q4_feasibility.py -q

只用已有 Torch reference 在内存中重放权重源准备；不调用 production GEMM/candidate。
没有新 GEMM/四模式/Event/NCU/MSE 数值，不重编译或修改正式 SM80/SM120 扩展。
结果目录与 SHA 回执随后补充；不能把旧最佳数据包装为此路线的结果。

# v103：无损结构化稀疏路线的数据可行性检查

状态：72 条完整权重检查通过。**停止“稀疏主项＋标量 SIMT 残差”路线**；
不是新 kernel，不计作性能提升，当前最佳与正式默认不变。

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
## 完整 24 样本结果

每列均为 24 样本中位数。百分比的分母分别是全体 Q4 权重、8 元素块、G128；
不能把“权重零值比例”当成符合硬件结构的比例。

|后端|Q4 零值比例|最低精确残差 / 全权重|需残差的 8 元素块|无需残差的 G128|标量残差有利模型 ms|
|---|---:|---:|---:|---:|---:|
|O3|31.0353%|24.5796%|94.0722%|0.0420%|0.866567|
|O7|28.1410%|26.2743%|96.1513%|0.0278%|0.926314|
|O8|21.3653%|31.0291%|98.4888%|0.0103%|1.093945|

残差比例的样本范围分别为 O3 9.0695–26.8855%、O7 9.6840–26.6334%、
O8 13.8872–31.4021%。不能只挑最稀疏的一个样本作为 24 组结论。
所有 48 个 O7/O8 weight source 与 v99 的完整 SHA 一致；O3 保存的 Q4 与当前映射一致。
256 种零位模式穷举证明每块残差非零个数最少；随机 INT8×Q4 整数点积验证主项与残差重构。

标量模型计算可复现为：

    residual_MACs = 4096 × residual_nonzero_weight_count
    t_ms = residual_MACs / (108 × 1410 × 1000 × 128)

当前参考主基准的 Event GEMM 约 O3 0.44 ms、O7/O8 0.47–0.48 ms（旧轮实测，不是本轮配对）。
**即使不计主项 MMA、地址、读取和 scale，仅标量残差已失去整体提速潜力**，
因而停止这一特定实现，不投入新 CUDA kernel、NCU、24 样本性能或四模式重复实验。

理想四 MAC vector instruction 的同一模型中位数为 0.216642/0.231579/0.273486 ms，
只是忽略 gather/packing/索引/依赖的容量值，**不能用标量模型否定所有 DP4A 或重排算法**。
当前约 94–98% 的 8 元素块仍需残差；向量方案也必须解决大量不规则激活取数。
本轮没有足够依据宣称它能显著超过已有最佳，因此不扩展为一次低把握优化扫描。
若未来引入新证据，应单独定义向量方案，不能把本次统计再包装为新迭代。

## 结论和证据边界

不改变原 quantization、scale、输入或 MSE 定义。没有新 GPU GEMM/MSE/转换耗时，
没有 sparse SASS 验收；只是排除固定 K 顺序下的标量残差路线，**不证明理论目标不可达**。
正式 A100 扩展 SHA 仍为：

    94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462

没有改动 5090、配置环境、锁频、等待 GPU 空闲或删除文件。
脚本/文档本地提交并推送后才同步 A100；服务器所有写入均在项目目录内。

原始 72 条统计、完整 source identity、环境/代码/扩展 SHA、模型假设和运行日志：
[v103 证据](evidence/a100_o378_roof_v103/README.md)。
目标尚未达到；该结果只增加明确的停止依据，不更新当前最佳性能表。

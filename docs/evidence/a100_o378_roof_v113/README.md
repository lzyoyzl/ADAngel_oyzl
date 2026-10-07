# v113：激活高位稀疏化的完整24样本证据

源 commit：`fb82df383007856d3b2bf6c558923d4bbf672061`，先推 GitHub，再在 A100
核验 bundle SHA、fetch/ff-only。只在项目目录生成报告，不重编译正式扩展，不改5090。

本次检查激活的 radix16 高位 H；不是重测 v103 的权重稀疏化或 v100 的稠密表示。
120条记录覆盖24样本×5策略，O7/O8的48份源格式 identity与v99完全相同。
原始/准备后 trace 深度哈希校验通过，保留固定 K/G128、全部非零残差与原 scale。

五种策略的全24投资门槛均失败；O8 balanced只有首层的4个样本单独通过。
这是标量残差补回的投入筛选，不是任意算法不可能加速的证明。
没有实现或运行候选 GEMM，没有新性能、MSE或GPU安全验收。
源格式参考量化仍使用已有 PyTorch GPU 操作，不能表述成“没有运行任何GPU操作”。

原始环境、120条逐样本数据、summary和完整日志见 `reports/`；
[index.json](index.json) 冻结所有可读产物SHA，测试重放汇总、来源与门槛。
完整归档（本地和A100）`tmp/o378_v113_activation_high_complete.tar.gz`，SHA256：
`f0d22fbd62de5fa90c896ad6a81521cd1c2e06b565ac5dfc2da1c085f15ea895`。

正式扩展前后SHA相同：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
参见[查重、预设预算与停止理由](../../o3_o7_o8_activation_high_sparsity_20261007.md)。

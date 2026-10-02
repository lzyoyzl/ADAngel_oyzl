# v56：O7/O8 全 K 整数累加的低成本可行性筛选

**决定：不直接将 O3 全 K 方案移植到 O7/O8；本轮不编译新 GEMM，不改变默认。**
这是一轮范围与工作量筛选，不是新的性能、MSE或安全验收。原生扩展 SHA-256 未变：
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

## 检查方法

在 A100 上使用原始 FP16 trace 的 `layer_00_{q,k,v,o}_proj` 四组样本，
按当前 `mixed_formats.quantize_source` 在内存生成完全相同规则的源格式，检查其 scale。
raw/prepared manifest 的关联及文件 SHA-256 验证通过；未改动或覆盖数据。
源代码 `3378ca3`；完整记录见 [feasibility.json](reports/o378_roof_v56/feasibility.json)。

实数意义下，O7 的公共因子为 `4*T_W`，组内剩余 `UE8M0_A*E4M3_W`；
O8 公共因子为 `T_A/4`，剩余 `E4M3_A*E6M2_W`。
将剩余 scale 精确表示成 `c_g*2^e_g`，其中 E4M3 的奇数尾数不超过15，
本项目 HiF4 E6M2 的奇数尾数不超过7；零块单独处理。
注意：抽出公共因子会改变现有 FP32 scale 的舍入点，不能声称输出逐位等价。

当前量化格式可达到的 payload 最大幅值给出组内点积界：

- O7：`B = 128*112*6 = 86,016`。
- O8：`B = 128*30*7 = 26,880`。

对每个输出位置，检查 `B*sum(c_g*2^(e_g-h)) <= INT32_MAX`。
它保证任意累加前缀的整数范围安全。分别采用最紧的逐输出最小指数锚点，
以及容易复用的“行最小指数＋列最小指数”锚点。
GPU 检查器的移位经过有证明的饱和限幅，不会令检查本身溢出，也不会改变通过/不通过判定。
637个合法scale编码穷举精确重构，随机 CPU/GPU 检查与 Python 任意精度整数对照均通过；
4项单元测试通过，见 [日志](reports/o378_roof_v56/unit_tests.log)。

## 结果

每个后端覆盖四组 `4096²`，共67,108,864个输出位置。

|后端|逐输出最紧锚点：界通过比例|行列可分离锚点：界通过比例|整张矩阵全部通过|
|---|---:|---:|---:|
|O7|97.7021%|95.3656%|0/4|
|O8|94.9272%|87.5395%|0/4|

**不通过表示这个保守界不能保证安全，不等于实际 partial 已经溢出。**
更紧的、依赖实际 payload 的界可能让更多位置通过，但它需要额外预处理；本轮没有测它。
因此这些数据也不能证明所有整数融合方案不可行。

## 为什么本轮不继续编译

1. 直接沿用“全矩阵guard，不满足则回退”的方案，四组样本都会整体回退。
   若细化为输出/CTA级选择，需要额外范围检查、元数据和混合执行，不能当作免费操作。
2. 相比 O3 的纯指数移位，O7/O8 还需处理整数尾数乘法，O8 还含行列尾数乘积。
   省去 I2F/FMA 不代表总指令或寄存器更少；不能以减少浮点指令数直接推断加速。
3. 更简单的 O3 全 K 实测也仅约 **+0.65%**，且增加 spill。
   在剩余实验预算有限时，当前证据不支持优先投入更复杂的直接移植。

同时核对了已有 v4/v5：O7 的幂次scale指数位替换已经测过，分别退化或没有确认收益，
不重复测试。后续若再研究整数融合，必须先提供更便宜的范围保证/对齐方法和真实工作量减少依据。

本轮 **无新增 GEMM 吞吐提升数值、无新增 MSE、无 conversion/Cold/steady 测量**。
已有最佳完整组合仍是 O3 GEMM54+转换2、O7/O8 GEMM59+转换5；不是把未运行的新候选记为0%提升。
整个检查约7.97秒只是检查器墙钟时间，不是实验 kernel 延迟。停止在四样本筛选阶段，没有伪称24样本验收。

复现（输出目录必须不存在）：

```bash
PYTHONPATH=python python -m unittest discover -s tests/unit \
  -p test_o78_integer_alignment_feasibility.py -v
PYTHONPATH=python python scripts/inspect_o78_integer_alignment_feasibility.py \
  --samples 4 --output reports/o378_roof_v56_recheck
```

归档 `tmp/o378_roof_v56_evidence.tgz` SHA-256：
`cea6335233fd290c3c5c166e7b538b8a5000f956204ebea6b210c1b98cd5a5d8`。
代码同步依次经过本地提交、GitHub push、A100 bundle fetch/ff-only merge；未改5090后端。

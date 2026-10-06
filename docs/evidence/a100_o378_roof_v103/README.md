# v103 原始数据：无损 paired-4:8 Q4 可行性

不是 GPU GEMM/性能实验，不包含新 MSE、Event、NCU 或 sparse SASS。
本轮未修改 CUDA kernel、正式扩展或默认。
只重放既有 source reference、核对 SHA，并统计完整 24×3 权重的最少精确残差。

- 实现 commit：d76a0510e8bb6ad324d21e909f92be53fce1bf78；
  GitHub 推送成功后，A100 通过同 SHA bundle fetch、ff-only merge，同步后运行。
- 初始 bundle SHA-256：
  7a145964e57c03a9ba0dc18c2ef5452951b1fab8609ad8f0e75c029c64b3c81e。
- 下载归档：项目 tmp/o378_v103_complete.tar.gz；
  服务器/本地 SHA-256 均为
  d5632e6c0d2bf6b4adcfe4ace7168026a5f42fd2240f0aed1c291bdad541a3e1。
- 未改原始 trace、prepared 文件、定点方法、G128 scale 或 5090。
- 正式 SM80 扩展仍为
  94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462。

文件：

- [72 条完整统计](reports/o378_roof_v103_sparse_feasibility/results.jsonl)：
  每样本每后端的零值/active pair/残差 histogram、完整 source identity 和 Q4 SHA。
- [汇总](reports/o378_roof_v103_sparse_feasibility/summary.json)：
  24 样本统计中位数和范围、声明假设下的残差工作模型。
- [环境与代码 SHA](reports/o378_roof_v103_sparse_feasibility/environment.json)：
  原始/prepared manifest、v99 provenance、量化源码、分析源码和扩展身份。
- [完整运行日志](reports/o378_roof_v103_sparse_feasibility.log)：24 个样本全部通过。
- [A100 单元测试](reports/o378_roof_v103_unit_tests.log)：10 passed。

48 个 O7/O8 weight source 与 v99 原 GPU 准备结果逐字段/逐 SHA 相同，
O3 直接读取 prepared Q4，并核对当前 E2M1→Q4 映射。
数据结果支持停止“主项 sparse MMA＋标量 SIMT 残差”，不证明所有重排/向量策略不可行。
没有删去残差来换精度，没有实现或宣称新的原生 sparse Tensor Core 后端。
正式目标尚未达到，最佳版本保持不变。

解释与公式见[报告](../../o3_o7_o8_sparse_feasibility_20261007.md)。

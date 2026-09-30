# v10：直接调用未恢复旧基线，继续隔离三阶段路径

构建源码 `3866017548de35168504fc3f890b0239e3ee327c`，分析脚本同步到 `5986631`。A100、CUDA12.8。构建和45实例ISA审计成功，但与v8的12个旧基线SASS仍全部不同，性能测试未启动。

例如O7/O8正式 `adangel_sm80_split_grouped_major<128,256>` 的静态SASS指令数：v8为832，v9和v10均为19768。因此单独去掉新增prefetch lambda不足以恢复原有代码生成，不能用此二进制计算优化收益。没有本轮GPU数值、MSE或性能结论。

下一轮将 `o3_optimized.cuh` 完整恢复为v8/c971aff源码，三阶段实验独立至 `o3_pipeline_candidate.cuh`，只由16/17调用；先检查机器码控制，再运行GPU验收。

本目录保存build_source、构建日志、审计和精确SASS比较JSON。大PTX/SASS留在A100 `reports/o378_roof_v10/audit/`。传输归档SHA-256：`c5eb94c370ee99578b12381dd6350a34413309a09212688992023f69a08850bd`。

# ADAngel 文档索引

- [O0–O4 最终实现与正式实验结果](o0_o4_final_results_report.md)：当前五个 production 后端、24 样本性能、转换开销、MSE、验收结果与性能差异分析；不包含历史候选或消融数据。
- [O1–O4 Nsight Compute profiling](ncu_profiling.md)：单真实样本、单 production kernel 的 NCU 驱动、过滤器、分层采集命令和可比性规则。
- [O3/O4 NCU 性能瓶颈分析报告](o3_o4_ncu_bottleneck_report.md)：NCU 基础、compute、scheduler、memory、occupancy 与 SASS 证据，以及 O3/O4 慢于 O1/O2 的根因和优化优先级。
- [O3 达到 O1 一半吞吐目标的优化结论](o3_half_o1_optimization_report.md)：最新 production、24样本配对结果、NCU/SASS 根因、目标是否达成及可行边界。
- [O1/O3 当前后端优化与性能分析](o1_o3_optimization_and_performance_report.md)：只描述两个当前 production 的数值路径、已采用优化、24样本性能/MSE，以及 O3 的 INT8 IMMA+位操作 SASS 限制。
- [O0/O1/O2 后端实现、差异与实验测量说明](o0_o1_o2_backend_report.md)：面向汇报的后端原理、MSE、计时口径及 O1 寄存器 partial 拟议优化。
- [O3/O4 Split 与 Bitwise 后端实现报告](o3_o4_backend_report.md)：论文对齐语义、G128/Q4、tile、计时、MSE 与 RTX 5090 验收。
- [实验协议](experiment_protocol.md)
- [数据格式](data_format.md)
- [Trace 采集](trace_collection.md)
- [Miniconda 配置](miniconda_setup.md)
- [服务器依赖](server_dependencies.md)
- [MXFP4 scale layout](mxfp4_scale_layout.md)
- [本机验证记录](local_validation.md)

# v99：输出 streaming-store，完整配对结果

源码 `90dfbf7496919da1c34d095354eb51e982beb219`，运行接口
`c74a7465f82829fbef437f2dbce9a5c348775e23`。本地实现、GitHub push 后才同步 A100。
保留独立 cubin；没有重编译正式扩展、改默认、改 5090、锁频或操作其他用户任务。

这是最终输出 `__stcs` 微调，不重复 v37 输入 L1 或 v38 输入 L2 prefetch。
G128、量化、所有 scale/guard、两路原生 INT4、FP32 输出、CTA 和转换均不变。

|三轮全 24 配对|GEMM 配对吞吐变化|95% speedup CI|结果|
|---|---:|---|---|
|O3 v89→v99|+0.24%|[1.000000,1.011792]|未确认，不采纳|
|O7 v78→v99|+0.43%|[1.002222,1.004334]|微收益，独立四模式确认|
|O8 v78→v99|+0.22%|[1.002137,1.003219]|微收益，独立四模式确认|

O7/O8 第二次 24 样本四模式运行：GEMM +0.21%/+0.43%，Cold +0.19%/+0.18%，
steady +0.19%/+0.19%（O7 区间跨 1）；转换没有确认变化。保留 v99 正向微调及 v78 主基准，
不声称所有模式均加速或主要瓶颈已解决。共 816 条真实输出逐位一致、MSE 不变。
所有 CV/离群值及两次 run 都保留；不是挑最快值或以复测覆盖首测。

## 文件及校验

- `reports/o378_roof_v99_{o3,o78}_codegen/`：生成 header/cu、PTX、SASS、resource、
  nvdisasm liveness、编译日志及 SHA receipt；旧编码对照和同 entry 两路 INT4/copy/store 审计。
- `runs/o378_roof_v99_o3/`：144 条 compute-only 原始记录、实际 CUDA 资源查询及检查。
- `runs/o378_roof_v99_o78/`：288 条 compute-only 原始记录、source provenance、环境/构建及检查。
- `runs/o378_roof_v99_o78_four/`：384 条独立四模式记录；所有 warmup=1000、repeats=200、inner=100。
- `runs/o378_roof_v99_o78_memcheck/`：小 M/N、K4096 验证；日志 0 errors。
  不是 4096³ sanitizer，不含本轮 race/sync 检查；O3 本轮没有 sanitizer。
- `tmp/`：未加工运行/编译日志、有限 memcheck 日志与正式扩展 SHA。
- `reports/o378_v99_complete_analysis.json`：**规范分析**，从所有 raw timings 重新计算统计、
  配对比值、bootstrap、CV、MSE/输出及资源身份；绑定当前 analyzer 源码 SHA。
  `o378_v99_paired_analysis.json` 是初始缓存阶段分析，作为历史文件保留。

完整 entry O3 local 空间 16→8B，不能把零整数 hot-loop local 说成整个函数零 spill；
O7/O8 entry local 为 0。两组仍 168 GPR、128 线程、3 CTA/SM，64 MMA、16 LDSM/G128/warp。
源码只改 store 提示，编译器也改变辅助指令分配；本轮无新 NCU，不强作缓存污染因果解释。

## 完整原始归档

本地和 A100 项目 `tmp/` 均保留；下载后 SHA 一致。Git curated 文本排除 `.cubin/.so/.o`，
二进制在归档及本地 `tmp/o378_v99_extracted/` 中保留，未删除。

|归档|SHA-256|
|---|---|
|`o378_v99_compile_complete.tar.gz`|`ff44f26086ee7b62ee4ab19fe6174ce5d6ceb596aa95050c3cb09b0b84165327`|
|`o378_v99_runtime_complete.tar.gz`|`914aaf8025fa39b2f769293b88f1e36f6bc0c55337ac559fae9b15f5f8679181`|
|`o378_v99_four_complete.tar.gz`|`9eb2a634d2380220d5c2b3385b859b52de57beaf53df7dfc22f720240989c44c`|
|`o378_v99_safety_complete.tar.gz`|`324d294f47e94b5404aadce87ecad46a8967c54fa5463fd22d91f010fec29ef3`|

正式扩展 SHA 仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
有限 GPU 数值检查及本地 202 项 CPU 源码/编码/证据重算测试通过；CPU 测试不冒充 GPU 内存检查。
[完整方法、四模式及 MSE 报告](../../o3_o7_o8_output_streaming_20261006.md)。

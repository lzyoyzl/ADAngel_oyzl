# A100 O3/O5/O6 合成输入 NCU 诊断

2026-09-29；源码入口 `8418cf6`，二进制为 `7760894` 构建，SHA-256：
`45f70a504a68295ff462d8bf79b989d68a6b1161ca0c53d4dabf8146d737a8e8`。

这是优化前 row-major scale 的诊断，不是 24 个真实 trace 的验收。
一个固定 Gaussian FP16 合成输入，4096³，seed=20260929。
O5/O6 CTA 为 64×128×256；每次只采集一个正式 GEMM，不包含转换。
NCU 2025.1.1，kernel replay，clock-control/cache-control 均为 none。
9 个 section：SpeedOfLight、ComputeWorkloadAnalysis、LaunchStats、Occupancy、
SchedulerStats、WarpStateStats、MemoryWorkloadAnalysis_Tables、InstructionStats、
SourceCounters；不是 `--set full`。

`reports/ncu/mixed_g128_v2/` 保存 capture、details、raw CSV、source SASS。
`runs/ncu_mixed_*_v2/` 保存源码/二进制 hash、输入和精确 launch 配置。
完整 `.ncu-rep` 保留于 A100 `/home/zlouyang/ADAngel_oyzl/reports/ncu/mixed_g128_v2/`。
NCU 重放时长不可当作普通 CUDA Event 性能，采样比例不等于耗时占比。

系统临时目录锁文件曾导致首次 profile 失败；未删除或改权限。确认没有其他
NCU 进程后，采用 NVIDIA 文档支持的 `TMPDIR=/home/zlouyang/tmp` 重新运行。
详见 [Profiling Guide FAQ](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)。

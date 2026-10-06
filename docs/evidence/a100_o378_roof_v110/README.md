# v110：64 accumulator / 八链的两条供数容量诊断

不是正式 GEMM 优化、真实 trace 延迟、MSE 或 kernel peak。完整说明见
[诊断报告](../../o3_o7_o8_capacity_shells_20261007.md)。

|目录|来源提交|状态|
|---|---|---|
|`reports/o378_roof_v110_shell_capacity`|`630dfe4e704e326bae292291fa2859e29f1962a2`|首次诊断编译，返回引用警告；未运行|
|`reports/o378_roof_v110_shell_capacity_r2`|`9ce9cdd1c7379e69cc7e224e8be3c67d47fb0ed2`|按值 CuTe view 修复；scaled 热 spill，未运行|
|`reports/o378_roof_v110_shell_capacity_r3`|`e86e28b3a63b06357b423239cbd5e82ee77a6f94`|简化合成 factor 初始化；scaled gate 仍失败|
|`reports/o378_roof_v110_two_shells`|`637434204bd7081c2f14512f0128ce94a478b4a7`|只复用 r3 cubin 的两个合格 entry；host 驱动单独编译|

前三个目录保留 source/artifact SHA、生成头、PTX、SASS、liveness、资源、build 日志；
顶层 `.log` 同样保留。修复不是新增优化迭代，失败 gate 从未改为通过。
第四个目录保留 3 轮×2路径×200 的全部 Event 值、CPU 全输出 checksum 和频率快照。
mode=2 从未启动；所有正式 extension SHA 相同。

结果按256组工作量除8归一化：寄存器路径0.293120 ms，shared路径0.295552 ms。
所有6组CV<3%，12次计时前及6次轮后逐位校验通过；不声称随机/真实样本或sanitize通过。
不能把不同编译路径之差精确当作LDSM独立成本，不能与历史真实延迟拼接成配对加速。

文本证据进入 Git；cubin/host executable 仅留在完整本地及服务器归档，不加入 Git。
冻结证据测试排除未提交二进制，其他所有源/产物 SHA 与原始记录逐项复核。

归档：`tmp/o378_v110_capacity_complete.tar.gz`，SHA256：
`5c23b7c21ba1a2c519c2442585b956a380c02d7b386a9c1fa782f5d53519c4fc`。

本轮不更新真实24样本延迟/MSE、默认、量化/转换或5090；目标未达到。

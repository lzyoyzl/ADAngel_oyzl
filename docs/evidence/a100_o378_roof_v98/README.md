# v98：固定八warp全K整数编译筛选（无GPU性能/MSE）

源码提交`ba5a927dec02b1685813581dfeee326de9808cb1`，先本地/GitHub，后A100项目内fetch/ff-only。
仅编译两个固定候选，无参数扫描、默认切换或5090改动。

|整数热循环静态工作|O3旧 → v98|O7/O8旧 → v98|
|---|---:|---:|
|分配寄存器 / 线程|168 → 121|168 → 128|
|静态指令 / warp|323 → 194|383 → 213|
|归一化指令增量（CTA warp4→8）|+20.12%|+11.23%|
|原生S4/S4 + U4/S4 MMA / warp|32+32 → 16+16|32+32 → 16+16|
|LDSM / warp|16 → 12|16 → 12|
|热循环local读写|0 → 0|0 → 0|

每CTA必要MMA总工作不变；LDSM总工作+50%。两组超过预设10%归一化指令预算。
旧完整编码对照、同entry PTX/SASS原生INT4与cg copy、Host CuTe坐标通过。
不投入候选GPU输出/MSE/Event/NCU/资源查询或安全检查，不从静态计数推断实测加速。

`reports/o378_roof_v98_{o3,o78}_codegen/`保留未加工文本及SHA receipt：
生成cu/cuh、PTX、SASS、resource、nvdisasm存活、编译/Host坐标日志和codegen.json。
`tmp/`保留150项CPU/source/历史证据测试日志、编译命令输出和正式扩展SHA。
CPU测试不是候选GPU正确性；`.minnctapersm 2`不是Driver occupancy实测。

完整原始归档（包括cubin和Host坐标可执行文件）在本地及A100项目
`tmp/o378_v98_compile_complete.tar.gz`，大小约2MiB，SHA-256：
`fd52c62bf9e6ee4503f2d668c80a74f75492cfd5a38aa9b0e219e5ffc66f5a31`。
二进制不作为curated Git文本证据；本地`tmp/o378_v98_extracted`也保留完整提取。

正式扩展未重建，SHA不变：
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
当前最佳仍O3 v89、O7/O8 v78+v73；本轮没有新性能或MSE。
完整说明：[v98报告](../../o3_o7_o8_eight_warp_fullk_20261006.md)。

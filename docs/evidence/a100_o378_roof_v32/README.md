# v32：M32/N128 tile 预检（尚未上机，不计入最佳结果）

候选57/58基于v31的异步FP32 scale搬运，CTA改为`32×128×128`，
4 warp按WM1/WN4分布。保持每warp的M32，使B fragment仍跨两个M atom复用。
两/三阶段、原生U4×S4与S4×S4、G128独立scale和升序FP32 FMA不变。

目的：每线程输出accumulator从64降至32，在不强制大量spill的情况下争取4 CTA/SM。
本地CUDA12.5独立编译：57/58均128寄存器、0 stack、0 spill。
这是资源预检，不是CUDA12.8结果、GPU正确性或性能证据。
已接入host dispatch、setup及prepared-core实验调度；尚未通过A100验证，正式默认及v31对照不变。
四模式入口明确拒绝57/58，先筛选GEMM收益，胜出后再接入端到端验收，禁止误用旧M64 grid。

本地host-only CuTe检查使用实际候选Config：128线程各拥有32个输出元素，
32×128共4096个坐标均有且只有一个owner，低/高INT4 fragment坐标一致；
两/三stage Storage分别25856/38784字节，各stage的A/W scale地址均16B对齐。
检查源码：`tests/cuda/validate_roof_m32_coordinates.cu`；无GPU launch，不能替代硬件验证。

与旧v26/N64候选不同：本轮缩小M，保留N128，WM1/WN4而非WM2/WN2，
同时使用v31异步scale。不能把v26负结果直接当作本轮结果，也不能预先假定会胜出。

代价必须一起核对：当前每G128/CTA读A低高共8KiB、W8KiB；新tile读A4KiB、W8KiB，
CTA数翻倍，总payload请求量预估增加50%。M方向更细造成更多W重复消费；
缓存可能合并请求，不能据此声称HBM流量也增加50%。
数学MMA/I2F/FMUL/FMA工作不减少，LDSM和CTA管理工作可能增加。

下一步需验证host grid/元数据的M维一致、16B scale对齐、CuTe输出坐标及M32边界，
再做原生INT4审计、逐位正确性/有限安全测试、与56的同轮性能/MSE配对。
若只降低寄存器却增加总延迟，记录负结果并保留56，不追求表面零spill。
不改变INT4路径、定点格式、跨G128 FP32累加或转换计时口径。

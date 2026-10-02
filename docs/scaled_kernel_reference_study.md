# 带 scale 的开源 kernel：与 O3/O7/O8 的对应关系

核对日期：2026-10-02。以下为源码研究，不是新增性能迭代；没有更改正式后端、量化或依赖。

## 最接近的结构

重点不是“有 scale”即可，而是 **scale 是否随 K 分组变化，以及乘在 MMA 前还是 partial 后**。
O7/O8 的核心是 `sum_g(P[g] * S_A[row,g] * S_W[col,g])`；
源格式的多级 scale 已合成为有效 scale，但组间差异仍须保留。
O3 的 A scale 不随组变化，可在最后恢复；W scale 仍逐 G128 保留。
全 K 整数候选只是精确重组这些权重 scale，不是删除 scale。

|参考实现|scale 在哪里应用|与本项目的关系|
|---|---|---|
|DeepGEMM SM90 FP8 1d1d|每128个K完成 MMA 后，将 partial 乘 A/B scale 并加入最终 accumulator|**最接近 O7/O8 的双侧逐组缩放**|
|CUTLASS SM90 FP8 blockwise scaling|独立临时 accumulator，按 scale 粒度提升/缩放到最终 accumulator|最适合对照 fragment 坐标、scale 广播、临时量生命周期|
|QServe / OmniServe W4A8 per-group|带整数 scale/zero 的 INT4 权重先恢复成 INT8；再 INT8 MMA，最终恢复外层 scale|可借鉴多级 scale 分工与布局；不能直接替换两路原生 INT4|
|Marlin grouped W4A16|组 scale 乘到解包后的 FP16 权重 fragment，再做 FP16 MMA|可参考 scale 与权重一起供数、寄存器双缓冲；数学路径不同|
|Triton block-scaled matmul 教程|MXFP4/MXFP8/NVFP4 的 scale 随 payload 输入原生 scaled MMA|用于比较 scale 预排布；不是 A100 软件逐 G128 缩放的替代实现|

逐张量/逐行列、仅在 epilogue 乘一次 scale 的 GEMM，不足以解释我们逐 G128 的开销。
硬件原生 block-scaled MMA 又是另一类，不能将其 scale 成本直接当作 A100 软件缩放成本。

## 已检查的具体源码

### 1. DeepGEMM：优先级最高

[sm90_fp8_gemm_1d1d.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh)
明确约束 BLOCK_K=128。SFA/SFB 与 A/B 共用 stage 完成条件；consumer 先取得 scale，
执行并等待本组 WGMMA，释放 stage 后，在寄存器里完成缩放累加。
代码分别保留本组 `accum` 和跨组 `final_accum`，A 的行 scale 与 B 的列 scale 在 fragment 内复用。

**借鉴点：scale 必须作为流水线数据处理；区分 shared 数据最后使用时刻和寄存器后处理结束时刻。**
但它使用 SM90 TMA/WGMMA 与动态寄存器分配，A100 不能照搬；
双侧G128数值结构相似，不意味着指令、占用率或延迟相同。

### 2. CUTLASS：对照通用实现

[SM90 blockwise collective](https://github.com/NVIDIA/cutlass/blob/db1c288993354c88e551c40c19a8fb93a774a241/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp)
是当前项目 pinned commit 中已有的源码，无需升级依赖。
它通过 `partition_C` 将 scale 映射到输出 fragment；显式区分临时 MMA 累加与最终 FP32 累加，
按 promotion interval 应用 scale，而非把所有 K 先加完再任意乘一个 scale。

值得核对我们的 scale 广播是否已有编译器复用、临时 partial 能否更早结束生命周期；
不能跨不同 G128 scale 直接增加 promotion interval。
[SM100 示例81](https://github.com/NVIDIA/cutlass/blob/db1c288993354c88e551c40c19a8fb93a774a241/examples/81_blackwell_gemm_blockwise/README.md)
也明确包含软件 block/group scaling，不能把所有 Blackwell 带 scale kernel 都归为硬件缩放。

### 3. QServe：多级 scale 有参考价值，但不是相同计算实验

[w4a8_per_group/gemm_cuda.cu](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/qgemm/w4a8_per_group/gemm_cuda.cu)
可直接看到 packed group scales/zeros、异步搬运和寄存器解包。
`share_to_reg_one_stage_B` 在整数域应用组参数，MMA 使用 `s8×s8→s32`。
因此它适合研究“外层与组内 scale 分开处理”和按消费顺序重排，
**不能将其 QoQ 量化或 INT8 MMA 偷换为我们当前双 INT4 实验**。

### 4. Marlin：scale 消费位置与生命周期

[marlin_cuda_kernel.cu](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu)
的 grouped 路径先解包、乘权重组 scale，再执行 FP16 MMA；只有非 grouped 路径才可把 scale 推迟到输出。
其权重/scale 布局、shared→register 双缓冲、缩短片段存活时间值得参考。
它主要面向 W4A16、小/中 batch；不能将其带宽型加速比外推到4096³的双INT4计算。
该链接为研究时的 master，不作为项目构建依赖；采用代码前还须固定提交并保留许可。

### 5. Triton：原生带 scale 格式的布局对照

[官方 block-scaled matmul 教程](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html)
提供 MXFP4、MXFP8、NVFP4 的完整 kernel。它将自然二维 scale 预排为消费时连续的块布局，
在主循环中加载 payload/scale，再交给 `tl.dot_scaled`；NVIDIA 路径依赖原生 block-scaled Tensor Core。
可借鉴“按消费者访问顺序存放 scale”，但不能把其硬件缩放成本或指令直接移植到 A100。
教程的格式与 scale 粒度不等于本项目的 G128 实验变体，不能为了使用它而改变我们的量化。

## 从参考源码到候选的筛选口径

|问题|在本项目中怎样核对|不能据此宣称什么|
|scale 是否随 K 变化|检查 `S_A[row,g]`、`S_W[col,g]` 的实际索引与消费点|不能把仅 epilogue scale 的 GEMM 当成同类性能上界|
|加载与计算是否重叠|对照 cp.async、stage 最后一次读取、barrier 与寄存器后处理|Hopper 的 TMA/WGMMA 方案不能原样用于 A100|
|临时量是否过多|检查 partial、A/B、scale 的共同存活范围及编译寄存器/spill|源码数组少了，不等于实际寄存器或延迟一定减少|
|已有优化是否重复|与当前 group-major、异步 scale、N64 流式片段等实现逐项比较|已经具备的技术不算新一轮优化收益|

例如这轮独立 interleaved-merge 编译候选仅尝试缩短 partial 生命周期：
四个独立 N atom 各用一组整数 fragment 先算 high、乘16、再算 low；保留原始两路 INT4、
各 G128 scale 和 FP32 顺序。这是针对本项目的推导，不声称来自上述项目的同款实现。
需要编译资源审计后才决定是否投入 GPU 测量；当前不因数学等价就宣称性能提升。

## 对下一轮的实际指导

当前实现**已经有** G128-major payload/scale、cp.async、swizzled shared layout、寄存器 partial、
部分流式 fragment 和向量转换，不应把这些成熟做法再次包装为新候选。
下一步先对照 DeepGEMM/CUTLASS 的“scale读取—MMA—stage释放—缩放累加”边界，
检查是否能在不增加大批常驻寄存器的情况下，让搬运/后处理重叠得更好。
这是待证实方向，不是已经找到的确定收益；已失败的 producer warp、整批预取挪动等不重复测试。

只选择一个能明确减少动态工作或缩短真实等待的新差异：先检查 SASS、寄存器和同步安全，
再进行少量配对初筛；有收益才扩大24样本、MSE和四模式验证。
保留独立 G128 scale、两路原生 INT4、FP32输出与原计时口径，不修改源量化来换取虚假加速。

补充：[LiquidGEMM 论文](https://arxiv.org/html/2509.01229v1)讨论 load/dequant/MMA 重叠，
但依赖 Hopper 与专门的量化方法。当前只核对论文，未确认作者的公开 kernel 仓库；
不把第三方重实现称为作者官方源码，也不直接采用其量化变更。

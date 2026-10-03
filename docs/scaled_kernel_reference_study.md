# 带 scale 的开源 kernel：与 O3/O7/O8 的对应关系

源码核对日期：2026-10-02；相关实测结果更新：2026-10-03。
源码研究本身不计作性能迭代，下面另列已完成的独立实测；正式后端、量化和依赖未更改。

## 参考后的实际效果：区分来源与实测收益

|参考或思路|本项目实际行动|已经验证的效果|
|---|---|---|
|CUTLASS SM80 分段异步拷贝|按现有布局实现 v64 分段 copy，与原 kernel 配对|O3/O7/O8 吞吐 −5.36%/−7.55%/−6.97%；不采纳|
|CUTLASS/CuTe fragment MMA遍历|v88以pinned CuTe整个fragment调用替代手写atom顺序|24样本O7/O8吞吐点估计+0.67%/+0.45%，两者区间跨1；未确认收益，不采纳|
|DeepGEMM / CUTLASS 逐 K-group scale|核对 partial、scale、stage 释放与最终累加的生命周期|已有结构可对照；没有可独立归因的新正收益|
|Marlin / QServe 解包与 scale 布局|研究向量解包、元数据复用与量化层级；不替换实验的两路 INT4|未直接移植其完整 kernel，不能引用其论文加速比作为本项目效果|
|向量化转换、减少中间重排（通用实践）|本项目 v54 conversion5|O7/O8 转换配对吞吐 +149.42%/+70.68%，Cold +13.06%/+8.39%；输出逐位不变|
|本项目精确 factor/整数范围推导|v67 全 K 整数累加独立候选|24样本×3轮 cached GEMM +6.84%/+7.04%；不是端到端收益|
|本项目减少重复payload读取|v69 将组平方和融合进转换|24样本 Cold +0.90%/+1.34%、steady +2.19%/+2.88%；仍有明显CV失败，默认不变|
|本项目减少metadata启动/读取|v73 将转换与行factor/anchor/范数生成融合，同一v67 GEMM|相对v69，24样本转换 +14.23%/+11.15%、Cold +1.83%/+1.66%；不是新增GEMM收益，直接计时CV仍大量失败|

最后四项是针对本项目数据格式实现的优化，**不声称直接来自某一个开源 kernel**。
v88参考落实到实际机器码，主循环383→377条、MMA复用标记25→27，但不足以证明稳定提速；
不把静态`.reuse`标记当作动态bank冲突改善。详见[v88证据](evidence/a100_o378_roof_v88/README.md)。
上述百分比来自各自同轮控制/候选配对，不能跨不同轮次相乘或拼接成累计实测。
带 scale 的开源实现主要帮助我们辨认哪些开销可以前移/复用、哪些量化语义不能省略；
是否有效仍以本项目的指令审计、MSE、安全性与配对计时为准。

后续NCU驱动的v81再次说明这一点：完整warp搬运A/W因子消除了目标excessive wavefronts，
但四样本O7/O8配对吞吐仅+0.57%/−0.57%，两者区间均未确认收益，已停止。
它是针对本项目实测热点的修改，不是新开源kernel移植，也不能用计数器改善替代延迟证据。
[v81结果](evidence/a100_o378_roof_v81/README.md)

v83进一步独立测试扩大N tile来复用A片段：相同输出工作量LDSM/copy条数下降25%/30%，
但寄存器168→255、驻留CTA 3→2，四样本O7/O8配对吞吐−0.90%/−1.36%，未采纳。
这也是本项目候选，不是直接移植某一开源kernel；供数复用与资源占用必须一起验收。
[v83结果](evidence/a100_o378_roof_v83/README.md)

v85进一步测试按MMA寄存器消费者顺序预排packed INT4，并以LDS.128替代LDSM。
修正CuTe N-warp坐标并通过正确性后，四样本O7/O8配对吞吐−2.65%/−2.91%，未采纳；
必要MMA和供数数量不变，168regs/3CTA未改善。它也是本项目独立候选，不是移植Marlin/QServe；
不能仅因使用“消费者导向布局”就宣称获得开源实现的加速效果。
[v85结果](evidence/a100_o378_roof_v85/README.md)

v86又用同一样本的full NCU检查v78/v85：LDSM替换为等量LDS，shared读取wavefront都为22.020096M，
168regs/3CTA不变，eligible和issue active下降。按消费者布局预排并未减少必要供数；
不因开源实践采用类似思想就继续扫描该方向。本轮是诊断，没有新的Event加速比。
[v86对照](evidence/a100_o378_roof_v86/README.md)

[v54 转换证据](evidence/a100_o378_roof_v54/README.md)、
[v64 分段 copy 负结果](evidence/a100_o378_roof_v64/README.md)、
[v67 缓存 GEMM 证据](evidence/a100_o378_roof_v67/README.md)、
[v69 完整在线路径证据](evidence/a100_o378_roof_v69/README.md)、
[v73 行级融合准备证据](evidence/a100_o378_roof_v73/README.md)。

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

### 6. CUTLASS SM80：A100 可迁移的拷贝调度

[mma_multistage.h](https://github.com/NVIDIA/cutlass/blob/db1c288993354c88e551c40c19a8fb93a774a241/include/cutlass/gemm/threadblock/mma_multistage.h)
的 `mac_loop_iter` 在 warp MMA 步骤之间调用 `copy_tiles_and_advance`，
将下一 stage 的 global→shared 拷贝分段发出，并使用双缓冲 shared→register fragment。
它不是本项目双侧 G128 scale 的现成实现，但比 SM90 TMA/WGMMA 更适合作为 A100 的调度参考。

据此设计独立 v64：将相同的 payload 拷贝拆成四段，分别穿插到前半段的 MMA 工作中；
scale 随第一段准备，最后一段才 commit，保留原来的 wait/barrier、CTA 和级数。
它不同于 v42 将整批 copy 整体挪动；不减少必要 MMA，也不改变数学。
这是按本项目布局重新实现的调度实验，不是复制 CUTLASS 源码或其性能结论。
编译先核对寄存器、spill、额外寻址和两路 INT4，再决定是否进行 GPU 初筛。
v64 实测已完成：虽消除 spill，四样本配对吞吐 O3/O7/O8 为−5.36%/−7.55%/−6.97%，
输出/MSE不变。停止候选，不扫描相邻位置；不能把参考项目的成功直接推定为本项目的收益。
[v64证据](evidence/a100_o378_roof_v64/README.md)

## 从参考源码到候选的筛选口径

|问题|在本项目中怎样核对|不能据此宣称什么|
|scale 是否随 K 变化|检查 `S_A[row,g]`、`S_W[col,g]` 的实际索引与消费点|不能把仅 epilogue scale 的 GEMM 当成同类性能上界|
|加载与计算是否重叠|对照 cp.async、stage 最后一次读取、barrier 与寄存器后处理|Hopper 的 TMA/WGMMA 方案不能原样用于 A100|
|临时量是否过多|检查 partial、A/B、scale 的共同存活范围及编译寄存器/spill|源码数组少了，不等于实际寄存器或延迟一定减少|
|已有优化是否重复|与当前 group-major、异步 scale、N64 流式片段等实现逐项比较|已经具备的技术不算新一轮优化收益|

例如这轮独立 interleaved-merge 编译候选仅尝试缩短 partial 生命周期：
四个独立 N atom 各用一组整数 fragment 先算 high、乘16、再算 low；保留原始两路 INT4、
各 G128 scale 和 FP32 顺序。这是针对本项目的推导，不声称来自上述项目的同款实现。
v63 已完成编译、正确性与四样本初筛：虽然 spill 降低，但三个后端均变慢，已停止；
不能因数学等价或 partial 数组更少就宣称性能提升。
[v63 实测证据](evidence/a100_o378_roof_v63/README.md)

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

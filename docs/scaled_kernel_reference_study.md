# 带 scale 的开源 kernel：与 O3/O7/O8 的对应关系

源码核对日期：2026-10-02；多级 scale 源码复核补充：2026-10-08。
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
后续当前结果和否定证据以[迭代台账](o3_o7_o8_iteration_summary.md)为准，不能将上表理解为最新最佳汇总。
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

### Tilus：借鉴布局与打包转换，不照搬 FP16 计算路径

核对用户提供的 arXiv:2504.12984v3（2025-08-31），重点为图1(c)、图9、§7.2、§8.1、§9.4。
同时核对 [NVIDIA/Tilus 固定提交4597cd5](https://github.com/NVIDIA/tilus/tree/4597cd5ba3f24501ba411cefa61a76fd9d3ba2c8)，
未安装Tilus或修改当前依赖。论文版本与当前源码分开记录，不假定二者代码完全相同。

其核心并非“任意低比特都能直接用原生Tensor Core”：先按MMA消费关系把权重tile预打包，
使global→shared→register加载后，每线程已经拥有所需bit；兼容的`View`只重新解释本线程位串。
数值`Cast`仍需执行，提前打包也有成本。图1(c)避免的是额外shared往返/跨线程布局转换，
不是移除所有shared memory，亦不是零成本反量化。

[实际grouped kernel](https://github.com/NVIDIA/tilus/blob/4597cd5ba3f24501ba411cefa61a76fd9d3ba2c8/examples/quantization/matmul_a16wx.py)
的`QuantizedMatmulChangeLayout`负责权重预排；主循环为
`load_shared → view(lowbit) → cast(FP16/BF16) → ×group_scale → dot`，
scale以`[K/group_size,N]`存放并参与异步流水线。
[cast emitter](https://github.com/NVIDIA/tilus/blob/4597cd5ba3f24501ba411cefa61a76fd9d3ba2c8/python/tilus/backends/emitters/cast.py)
使用LOP3/PRMT和向量数值修正，例如一次处理8个INT4；它没有提供本项目HiF4→Q4的同款数值转换。

| 可借鉴思想 | 与当前实现的关系 | 本轮处理 |
|---|---|---|
| 按MMA线程/元素所有权预打包 | 已有group-major、CuTe fragment和swizzled shared；v85/v86消费者预排改LDS.128已无收益 | 不重复该候选；只有发现可删除的实际重排/搬运才另做实验 |
| 寄存器内成组位操作，避免逐元素解包 | v118 NVFP4、v123 FP6已有；v138将HiF4 micro8/micro4共享字段纳入打包转换 | v138设计在本次阅读前已完成，不能把其收益追溯归因于Tilus；共同原则提供交叉验证 |
| 显式布局、内存层级和向量搬运 | 已有cp.async、寄存器partial、分阶段shared供数 | 审计实际SASS及生命周期，不把已有技术算成新优化 |

论文§9.4的prefill使用量化权重解码至FP16后的标准FP16 GEMM；图14也显示大batch收益与decode不同。
其主要低比特性能结论不能外推为本项目4096³双原生INT4的加速承诺。
本项目保留独立G128 scale、两路原生INT4、FP32输出与现有计时，不通过换FP16/INT8路径宣称达标。
截至本次源码核对，没有新增“Tilus带来的GEMM加速”测量结论。

### 本次补查：HiFloat4 与新版 Marlin 的多级 scale

[HiFloat4 官方 GPU 源码，固定提交 6d937b6](https://github.com/global-computing-consortium/HiFloat4/blob/6d937b6fcf34f63b8fc563bd72e3aea0f44a46b4/hif4_gpu/quant_cy/base/cusrc/hifxg_quant_cuda.cu)
的 `hifx_quant_cuda_inner` 先计算分层块参数，再按8元素/4元素的共享范围处理数据，
返回的是模拟量化后的浮点结果。README 的 GPU 示例随后调用普通 `torch.nn.functional.linear`。
**这是量化/反量化参考，不是可以直接移植的原生 HiF4 Tensor Core GEMM。**
本项目保留已确认的 G128 实验变体，不改成参考实现的默认分组。

[vLLM Marlin dequant.h，固定提交 87d9996](https://github.com/vllm-project/vllm/blob/87d9996abe7ae0173fa8c1b6ccc2411041bcac81/csrc/libtorch_stable/quantization/marlin/dequant.h)
将低比特解包表示为打包位操作加数值修正；FP4路径可把常量指数修正合并到后续 scale。
可借鉴的是**按字段共享范围复用元数据、保持打包形式、将固定修正并入已有步骤**，
而非照搬其FP16/FP8 MMA或更改本实验的两路原生INT4。

据此选择本项目独立 v138：保持 v123 FP6激活转换、v78 GEMM、G128 E6M2 scale和溢出guard不变，
只将HiF4权重的逐元素RNE替换为八nibble并行RNE；micro8/micro4各字节按16元素向量复用，
用标量DP4A计算精确平方和。它是本项目针对既有格式重新推导的代码，不是复制开源kernel，
也不是新的GEMM调度方案。与此前NVFP4/FP6打包转换不同，必须同时保留micro8/micro4两级局部指数。
编译和性能结果另见[本轮报告](o8_hif4_packed_conversion_20261008.md)。

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
截至v137，分段拷贝、提前读取、扩大tile、增加warp、系数预排、分离缩放、high/low拆链等
已有编译或完整配对否定证据；不能再次笼统地以“借鉴开源流水线”为名重复测试。
当前最佳GEMM已用带guard的全K整数累加，不能继续把旧版逐G128 FP32 FMA链当作它的主瓶颈。
先查台账及当前NCU，再决定是否存在确实不同、值得投入的机制。

只选择一个能明确减少动态工作或缩短真实等待的新差异：先检查 SASS、寄存器和同步安全，
直接按当前要求进行24样本×3轮配对、MSE和相关计时验证，不再做小规模性能初筛。
小规模正确性、边界和sanitizer检查不属于性能初筛。
保留独立 G128 scale、两路原生 INT4、FP32输出与原计时口径，不修改源量化来换取虚假加速。

补充：[LiquidGEMM 论文](https://arxiv.org/html/2509.01229v1)讨论 load/dequant/MMA 重叠，
但依赖 Hopper 与专门的量化方法。当前只核对论文，未确认作者的公开 kernel 仓库；
不把第三方重实现称为作者官方源码，也不直接采用其量化变更。

## 2026-10-08：收紧通用优化方向，并复核真实的加权交错

仍在O3/O7/O8代表格式上优化，不另建通用编译框架。新机制应由位宽、分组scale和
数据依赖决定，而非某个格式的码值、这24组数据的特殊分布或某个编译器的偶然指令选择。
必要的格式解码和ISA适配不等于应当消除所有硬件相关代码。

补查两篇原始论文后，没有找到可以直接替换本实验的同语义kernel：

|参考|可以借鉴的原则|本轮不能直接采用的部分|
|---|---|---|
|[APEX4 §3.2、§4](https://arxiv.org/html/2606.08761v1)|把scale计算需求和Tensor Core供给一起建模|混合粒度方案改变组量化；不能把G128改成per-channel来声称本实验加速|
|[LiquidGEMM §5.1–5.3](https://arxiv.org/html/2509.01229v1)|避免为了流水化而新增寄存器↔shared往返；保留consumer内的数据复用|Hopper/WGMMA与W4→INT8反量化路径不同；不能替换我们的两路原生INT4|

**进一步检查发现：当前最佳机器码已经交错了不同输出片段的MMA与整数加权。**
不能因为源码先写MMA、后写scale，就把二者描述成整个warp完全分离的两个阶段。
同一输出的RAW依赖仍存在；下面也不证明硬件上这些指令实际同时执行。

新增`inspect_scaled_partial_overlap.py`，沿最后一次MMA的四个INT32结果，经过行因子乘法，
追踪到最终accumulator的IMAD更新。每个G128必须恰好找到16条链×4个值=64次更新，
丢失、重复或无法解析的数据流立即报错。复用v96/v97归档的同构建控制与候选，不重新编译。

|机器码|旧半tile的32次加权更新中，排在下一半tile首条MMA之后的数量|旧加权排空前已排入的下一半tile MMA数|
|---|---:|---:|
|O3 当前v89控制|12|9|
|O3 v96 / v97|12 / 12|9 / 9|
|O7/O8 当前v78控制|16|7|
|O7/O8 v96|10|2|
|O7/O8 v97|0|0|

这是**静态指令交错计数，不是周期数、硬件in-flight数量或性能百分比**。
旧审计的“第9条链从第33条MMA开始”只回答MMA链排序，不能单独回答scale是否被交错。
补充逐值追踪后，v96没有增强该交错，v97在O7/O8反而完全排空旧加权再开始新片段。
因此不以“更早开始下一片段”之名重跑它们，也不因v97静态指令少6条便升级为性能候选。

这次结论进一步缩小下一候选的范围：必须证明**比现有交错更强的依赖安排，或实际减少必要工作**，
同时核算寄存器、供数和同步代价。已经有的cp.async、八链与scale/MMA交错不能再次算作新优化；
增加producer、shared partial交接、扩大tile等已测负结果也不因论文采用类似名字就重做。
目前没有足够证据承诺某个新增通用候选会提速；不为凑轮次启动无依据的GPU扫描。

这是一轮诊断与筛选工具改进，**不是新增kernel性能迭代**，没有新MSE、Event或NCU数据。
O3 v89、O7/O8 v78及最佳转换组合、正式默认与5090保持不变，原GEMM目标仍未完成。
可重放JSON见[evidence](evidence/o378_scale_overlap_reaudit_20261008/)，输入文件SHA写入各JSON。
回放命令示例：

```bash
python scripts/inspect_scaled_partial_overlap.py \
  --codegen docs/evidence/a100_o378_roof_v96/reports/o378_roof_v96_o78_codegen/codegen.json \
  --sass docs/evidence/a100_o378_roof_v96/reports/o378_roof_v96_o78_codegen/o78_interleaved_tail.sass \
  --symbols adangel_roof_o78_eight_chain_candidate adangel_roof_o78_interleaved_tail_candidate \
  --output reports/scale_overlap_fresh/o78.json
python -m pytest tests/unit/test_scaled_partial_overlap.py tests/unit/test_interleaved_tail_probe.py -q
```

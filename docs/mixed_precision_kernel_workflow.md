# 混合低精度 kernel 的分层工作流（阶段稿）

范围：总结现有 O3 与正在开发的 O5/O6 共用计算路径。
O5/O6 前端格式尚未确认，本文不是最终实验验收报告。

## Conversion：先证明数值，再组织数据

    FP16 原始 A/W
      → 指定浮点格式及 G128 scale/内部指数
      → 解码局部数值并对齐
      → RNE + clamp 得到 Q4 / Q8 / Q6
      → Q6 符号扩展至 INT8（仅 O6）
      → A 的 low-U4/high-S4 打包，W 的 S4 打包
      → 有效 FP32 A/W group scale

必须把三类行为分开：量化改变数值；对齐/转定点可能舍入或饱和；
纯 packing、补码拆分与符号扩展不应再改变数值。
共享 scale、内部 micro-exponent、定点二进制小数位应最终对应到
明确的“整数单位所代表的实数”。不允许只保留 payload 而丢弃元数据。

转换单测先覆盖所有有限编码、零、tie、边界、异常 scale，再看随机和真实输入。
MSE 至少区分原始 FP16→格式的损失、格式→定点的新增损失，以及
实际 CUDA 输出对定点语义参考的实现误差。三者不可用一个 MSE 混淆。

在格式定义完成后，正式转换计时仍采用批量摊销，端到端直接计时。
compute-only 的前处理必须提前完成；缓存静态 W 不等于免除在线 A 转换。
目前新接口只收 prepared integer 输入，尚不报告正式 conversion/cold/steady。

## Optimization：围绕有效数据复用，而不是只比较位宽

| 层级 | 共用实现 | 应观察的成本/约束 |
|---|---|---|
| 全局输入 | A_split[2M,K/2]、W_q4[N,K/2]、双侧 G128 scale | 读取合并、额外 scale 流量、格式变换是否计时 |
| Global→shared | cp.async 两级 pipeline，字节/半字节一致的 swizzle | 预取重叠、stage 重用安全、shared 容量 |
| Shared→register | CuTe LDSM layout；从真实 fragment 坐标定位输出 | bank conflict、fragment 重复装载 |
| Tensor Core | low U4×S4 和 high S4×S4；权重片段供两路复用 | 发射带宽、IMMA 依赖链、寄存器存活 |
| G128 后处理 | low+16×high；INT32→FP32；乘双侧 scale，FMA 累加 | 通用 ALU 指令、scale 广播、不得跨组后缩放 |
| 最终输出 | FP32 accumulator 留寄存器，最后向量写回一次 | 寄存器/spill 与并发 CTA 的取舍 |

K256 pipeline 是一次搬两个 G128，不是把两个 group 合成一个整数点积。
窄 N-slice fragment 降低中间 INT32 存活量；扩大整体 CTA 并不要求
同时把整个 CTA 的所有 partial 保存在寄存器。

不要将“越少指令”“零 spill”“更大 tile”各自作为绝对目标。
保持数值语义的前提下，以配对性能、原始计时分布、ISA 和内存安全联合选择。
对具有非二次幂 scale 的新场景，旧 O3 的指数位加法技巧不能直接复用。

## 交付验证顺序

1. 固定格式契约和独立 CPU 参考，明确 MSE 主参考 O0。
2. 小形状 GPU 正确性、padding/sign、逐组 scale 和旧路径回归。
3. 审计同一正式函数的 U4/S4 IMMA、cp.async、资源/spill。
4. Compute Sanitizer 检查边界、stage 生命周期与读写同步。
5. 固定真实 24 样本，同进程交错顺序对照 O0；保留所有原始计时。
6. 仅在瓶颈仍不明确时做针对性 NCU；profile 时间不混入普通 Event 性能。
7. 发布格式、代码/二进制 hash、MSE、四模式结果和异常说明。

这个顺序中任一未完成项必须标为未验收，不能因某个合成 GEMM 很快就跳过。

# 实验协议

## 控制变量

所有配置消费同一份已准备的 `A_int8/A_scale/W_mxfp4/W_scale`，计算
`Y=A@W.T`。矩阵尺寸、样本顺序、CUDA stream、预热次数和测量次数完全相同。
公共 FP16 trace 的初始量化不计入任何配置。

## 计时边界

- O0-W-dequant：packed E2M1 + UE8M0 → FP16 W。
- O0-A-dequant：INT8 × per-row scale → FP16 A。
- O0-GEMM：FP16×FP16，FP32 accumulate/output。
- O1-W-convert：E2M1 nibble → 精确的 `2*E2M1` INT8 基值。
- O1-GEMM：INT8 MMA、每 K32 rescale、FP32 累加的整体。
- O2-A-quant：反缩放、K32 amax、UE8M0、E2M1 RNE、packing 与 SFA 重排。
- O2-W-layout：自然顺序的 `W_scale[N,K/32]` → CUTLASS SFB physical layout；
  packed MXFP4 权重数值本身不转换。
- O2-GEMM：MXFP4 block-scaled MMA 与 FP32 累加。

`conversion_only` 分别测转换 kernel；`compute_only` 预先完成 A 量化、SFA 与 SFB
重排，只测 GEMM；`cold` 包含该 variant 的所有转换；`steady_state` 缓存静态权重
转换。对 O2，steady-state 因而只保留在线 A 量化/SFA 重排与 GEMM。

## 统计规则

每项预热 50 次、测量 200 次；汇总 median、P5、P95、IQR、CV。CV 超过 3% 的
数据判为不合格，需要在保持配置不变的前提下重测。

转换组件统一采用独立批量CUDA Event计时。O0-W、O0-A、O1-W、O2-W-layout和
O2-A在每个外层样本中连续执行
`timing.conversion_inner_repeats=100` 次，单次延迟取批量耗时除以100。
`conversion_only/total` 对该variant的完整转换序列采用相同批量摊销方法。

`compute_only/total`、`cold/total`、`steady_state/total` 保持单次直接路径计时：
cold只执行一次在线权重/激活转换，steady-state继续缓存静态权重转换。转换组件的
批量区间不进入端到端区间。分阶段批量均值与直接total独立测量，因此二者不要求精确
相加；端到端结果以direct total为准。每条结果记录通过`timing_method`明确保存
实际计时方法和inner repeats。转换吞吐以实际读写字节/时间报告；GEMM等效吞吐按
`2MNK/time`报告。
主表将 O2-W-layout 记录为 o2/weight_conversion；其吞吐按 2*N*(K/32) 字节计算。
O2-A 按 M*K + 4*M + M*K/2 + 3*M*(K/32) 字节计算；未触碰的物理布局 padding 不计入。

## 正确性

每个样本只保存一次 O0 FP32 输出。O1/O2 输出转 FP64 后计算 MSE，O0 对自身为
0。对 24 个 MSE 报告逐样本值、median、IQR、max 和 bootstrap 95% CI。

正式运行前执行一次指令审计。审计证明使用目标 Tensor Core 指令，但不进入主结果。

## O1 实现 A/B 与 CTA 消融

O1 的公开 production 实现由源码常量 `kProductionO1Implementation` 唯一指定。
寄存器候选未通过 RTX 5090 验收前，该值保持 `shared_partial`。内部接口
`adangel._sm120._benchmark_o1_impl` 只用于候选验证，正式 runner 不暴露实现选择。

A/B 固定使用同一进程、同一输入、单 CUDA stream，并按样本和 mode 交替实现调用
顺序。三种实现为：

- `shared_partial`：`64x32x32`，历史 production baseline；
- `register_64x32`：相同 CTA、TMA stage 和 producer/consumer 数，仅改变 partial 路径
  及为获得公开 fragment 坐标而必须采用的 CuTe m16n8k32 MMA API；
- `register_128x128`：`128x128x32`、1 producer/16 consumer，仅作为 M/N CTA 消融。

O2 保持 `128x128x256`，不得为追求表面 CTA 一致而修改。O1 每个 K32 后必须软件
缩放，O2 可在一个 K256 mainloop tile 内由原生 block-scaled MMA 消费8组 scale。

每条 O1 结果必须记录：

```text
implementation_key
partial_storage
shared_partial_redistribution
cta_tile
kernel_symbol
mma_api
mma_shape
production_selected
```

候选晋升要求：正确性/MSE 回归通过；所有正式计时阶段 CV<3%；24 样本
compute-only 配对加速比 bootstrap 95% CI 下界>1；PTX/SASS 同一正式 symbol 包含
TMA 与 signed INT8 MMA；SASS/resource usage 无 local-memory spill。旧 run 缺少上述
元数据时不能冒充寄存器后端结果。

`register_64x32` 是本轮必须验收的第一候选；`register_128x128` 仅用于 CTA 消融。
可选消融若出现 `LDL/STL`、非零 `STACK` 或 `LOCAL`，必须标记为 `DISQUALIFIED`，且
不得晋升，但不应阻断已经满足全部门槛的64×32候选。任何被选为production的实现都
必须通过无spill门槛。

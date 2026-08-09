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

每项预热 50 次、测量 200 次。每一次记录 CUDA Event 延迟；汇总 median、P5、
P95、IQR、CV。CV 超过 3% 的数据判为不合格，需要在保持配置不变的前提下重测。
转换吞吐以实际读写字节/时间报告；GEMM 等效吞吐按 `2MNK/time` 报告。

## 正确性

每个样本只保存一次 O0 FP32 输出。O1/O2 输出转 FP64 后计算 MSE，O0 对自身为
0。对 24 个 MSE 报告逐样本值、median、IQR、max 和 bootstrap 95% CI。

正式运行前执行一次指令审计。审计证明使用目标 Tensor Core 指令，但不进入主结果。

# O3 两路 INT4：补齐一次未运行候选的实测（v135）

本轮不是新的 CUDA 优化，也不是重跑已有负性能结果：复用 v115 的原始 cubin，
首次测量其完整24样本性能。O7/O8、正式默认和5090不变，不引入单路 INT8 Tensor Core。

旧 v115 将 high/low 两路原生 INT4 点积分开，再使用 PRMT+DP2A 进行精确整数合并：
`acc += low_dot*f + high_dot*(16*f)`。DP2A 是普通整数指令，不是 INT8 Tensor Core。
独立点积缩短合并前的依赖，但增加 packing、地址工作以及3条热 local 读取。
原编译 gate（340/323条静态指令、3条local）仍明确为失败，不覆盖或改写旧记录。

此前的5%工作量门槛是投入筛选，不是性能定理。依据用户已允许少量spill的条件，
这次只对**相同冻结二进制**做一次运行复核，不调整寄存器、布局或邻近参数。
保留原始scale、整数范围guard及两种回退；只有原guard安全且factor≤15的N128 CTA使用DP2A。
预计覆盖率来自旧数据，实际覆盖必须由运行元数据重新统计。

流程：原始PTX/SASS与SHA回放 → 实际3 CTA/SM容量 → 数值/边界/三路径混合验证 →
24样本×3轮、warmup1000、repeats200的交错配对、MSE及全部CV。
compute-only有明确收益才扩四模式；新增metadata打包计入W转换/Cold，compute和steady缓存。
无收益即结束这条实测路线。不能把静态指令变化称为加速，也不能将编译通过当作正确性通过。

```bash
TMPDIR="$PWD/tmp" python scripts/benchmark_o3_dp2a.py \
  --output runs/o378_roof_v135_full24 --samples 24 \
  --rounds 3 --warmup 1000 --repeats 200 --inner 100 --modes compute_only
```

首次运行已确认两端实际均为3 CTA/SM。验证脚本的旧计时断言只接受两个W准备kernel，
在新增metadata打包（实际第三个kernel）时终止；尚未进入真实样本性能测试。
修复仅更新本轮计时契约检查，保留第三个kernel的真实开销与首次失败日志，不改CUDA二进制。
当前状态：等待修复后完整验证与计时。

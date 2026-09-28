# A100 prepared split core：首轮证据

范围仅为 O5/O6 共用的已准备整数 GEMM，不代表完整 O5/O6 验收。

源码：3bc6111315ec7f9dee4ad406cc47254147ba0e5e。
二进制 SHA-256：
f7398ddfdf595d5d32f2f8ec26cbb158625e2478e9a632314e03d009cfa801ce。

- validation.json：56 项小规模/边界/旧 O3 回归，另有两项 4096³ 合成输入测试；
  七类非法输入被拒绝。两项 4096³ 输出对独立整数语义参考的 MSE 均为 0。
- audit.json：两种 tile 同一函数 PTX/SASS 均有 U4×S4、S4×S4 MMA 和
  cp.async/LDGSTS，无 INT8 退化。
- resources.txt：原始 cuobjdump resource 输出（含扩展内其他函数）。
- memcheck_grouped_v1.log：0 errors。
- racecheck_grouped_v1.log：0 hazards / 0 errors / 0 warnings。
- build_grouped_v1.log：CUDA 12.8 的完整构建日志。

| Tile | Synthetic GEMM median ms | Mean ms | CV % | 资源 |
|---|---:|---:|---:|---|
| 64×64×128 | 0.796672 | 0.792822 | 1.661 | REG90、STACK0、无 LDL/STL |
| 64×128×256 | 0.631808 | 0.620692 | 3.935 | REG128、STACK8、2 LDL / 2 STL 静态指令 |

大 tile 超过 3% CV，应标记为计时不稳定，不删除样本。小量 spill 按既有
用户政策允许保留，但严格零 spill 审计仍为 false。静态指令数不是动态次数。
validation.json 中 passed=true 是正确性通过，不是性能达标；
后续验证脚本已单列 correctness_passed 和 timing_stable_cv3 避免歧义。

这些输入直接构造整数和非二次幂 G128 scale，不经过 NVFP4/MXFP8/HiF4/NVFP6。
因此 MSE=0 只表示整数 kernel 数值正确，不是相对 O0 的量化误差为 0。
没有同进程 O0 配对，本表也不能用于宣称 O5/O6 已超过 O0。
全部 raw_ms 保留在 validation.json。

完整 PTX/SASS 原件位于 A100：
/home/zlouyang/ADAngel_oyzl/reports/audit_split_grouped_v1/。

# v119：纯寄存器 MMA 的编译依赖候选

本轮回到 GEMM，只检查一个此前没有测试过的编译层方向；不是再扫描 tile、stage、warp 或累加链数量。
正式默认、源格式、转换、G128 scale、FP32 输出和5090不改。

## 查重与假设

v63/v78 改 partial 合并方式，v92/v96/v97 改源码链调度或槽位；v117 改固定 high×16 的算术路由。
本轮都不重复。仅复制 pinned CuTe 的两个 m16n8k64 INT4 operation，将其中 `asm volatile` 改为 `asm`，
保留全部输入输出约束、指令字符串、寄存器类型和原 CuTe thread/value traits。
第三方源码不修改，cp.async、ldmatrix、barrier 和 fallback 仍是原实现。

依据 [NVIDIA CUDA 12.8 inline PTX 文档](https://docs.nvidia.com/cuda/archive/12.8.0/inline-ptx-assembly/index.html#incorrect-optimization)，
非 volatile asm 按输出依赖优化，volatile 限制 PTX 生成阶段的移动/删除。假设是取消纯寄存器数学操作的
额外编译顺序约束，可能改善标量与 MMA 工作交错；这不是保证，并不意味着可以删除 warp collective 的参与约束。
固定 loop 必须完整 warp 参与，数据依赖不删；只有通过机器码门槛后才进行 GPU 正确性/同步和完整24样本配对验收。

## 预先规定的投入门槛

固定 O7/O8 的64×128×128、四warp、两stage、八条 partial 链、原 guard 和 epilogue。
原v78正式同名入口保留，必须与旧CUBIN完整编码一致。候选同entry必须包含两路原生INT4及cg async copy。

- 同整数循环仍为32 S4/S4 +32 U4/S4 MMA、16 LDSM、10 async copy，barrier不变；
- 分配寄存器≤168，热循环 local load/store=0，循环总工作不增加超过2%；
- 同时至少减少5%静态循环指令，或分配寄存器降至≤128；寄存器阈值不是已证明4CTA容量；
- 未达门槛则不启动候选性能测试、不扫描相邻 asm 变体、不迁移O3；
- 所有静态证据只说明编译潜力，不是性能、MSE或GPU安全结果。

```bash
python -m pytest tests/unit/test_register_mma_codegen.py -q
python scripts/probe_register_mma_codegen.py --output reports/o378_roof_v119_codegen
```

实现先在本地提交并推送，再在A100项目内同步和编译。GEMM接近有效吞吐上界的目标不变。

## A100编译结果与决定

原始运行源码commit `e92177f5`，CUDA12.8.93、固定CUTLASS commit，控制入口与原v78完整编码一致。
候选与控制入口的**全部SASS指令编码也完全相同**，不只是指令统计相同。

| 同entry静态指标，不是实测延迟 | 原v78 | 新候选 |
|---|---:|---:|
| 整数循环指令 | 383 | 383 |
| entry分配寄存器 / stack bytes | 168 / 0 | 168 / 0 |
| 整数循环活跃GPR峰值 | 166 | 166 |
| S4/S4 + U4/S4 MMA | 32 + 32 | 32 + 32 |
| LDSM / async copy / CTA barrier | 16 / 10 / 1 | 16 / 10 / 1 |
| 整数热循环local load/store | 0 / 0 | 0 / 0 |
| 静态启动未完成链峰值（非硬件并发） | 8 | 8 |

取消编译层volatile没有落实为执行代码的变化，因此不可能将再次计时的频率/调度差异归因于这个优化。
预先规定的潜力门槛失败，**停止该路线**，不扫描其他asm变体、不迁移O3，不启动候选GPU/性能/MSE/NCU。
没有新增实测提升数字；O3 v89、O7/O8 v78的最佳GEMM及原MSE保持原有依据，正式默认不改。

14份原始文本和BSD注记、SHA及CPU重放已冻结。后续仅给未来生成的CuTe副本保留完整上游版权注记；
原测试commit/header/raw记录不改，不为注记修复重复编译或进行优化重测。
[原始证据与复算方法](evidence/a100_o378_roof_v119/README.md)。主目标尚未达到。

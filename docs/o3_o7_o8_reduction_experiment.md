# O3/O7/O8：独立 FP32 累加链实验

状态：用户已允许独立测试；不替换任何正式默认。只在 A100/SM80 构建。

## 改什么、不改什么

原来每个输出按 G128 顺序更新一个 FP32 accumulator。候选24把偶数、奇数
group分别放入两个累加器；候选25按 group编号模4分到四个累加器，最后树形相加。
每条链仍执行 FP32 FMA，最终合并使用显式 `__fadd_rn`。

保留原量化、两路原生 U4×S4/S4×S4 MMA、整数 partial 重构和每个 G128 独立 scale。
CTA沿用候选6的64×128×256、256线程、两级cp.async，便于隔离比较。
不做跨CTA Split-K、不把所有group的中间矩阵落盘，也不宣称所有乘法同时执行。

这是**改变浮点求和顺序**，不是逐位等价优化。潜在收益是缩短同一输出的依赖链；
代价是每线程输出累加寄存器由32个增加为64/128个，以及最终加法。
是否能更快取决于编译后的调度、spill和驻留warp，不能由链长直接推断加速比。

## 独立验收

- 旧候选继续要求与原正式实现逐位一致；24/25必须显式传入 `--allow-reassociation`。
- 记录改变的输出元素数、相对旧输出的最大差值/MSE、相对FP64语义参考的误差。
  FP64参考从相同packed整数重建，使用相同的已舍入FP32 group scale乘积。
- 语义参考比较 `rtol=1e-3, atol=1e-3`；24真实样本重新计算对O0和O5/O6的MSE。
  MSE回归以 `rtol=1e-5, atol=1e-12` 单独标记，不能沿用旧MSE或隐藏失败。
- 同进程、同输入、循环交错测量，保留全部原始计时及CV失败；四种计时定义不变。
- 同entry检查两路INT4 SASS、异步拷贝、I2F/FFMA/最终FADD；报告spill，运行内存/同步检查。
- 对旧正式kernel和旧候选比较机器码，避免新编译改变对照；不通过则停止性能归因。

## 命令

重新编译并审计后，在A100项目目录执行（输出目录必须全新）：

```bash
python scripts/benchmark_a100_roof_candidates.py --synthetic --validate \
  --allow-reassociation --tunes -1 6 24 25 --rounds 8 \
  --output runs/o378_reduction_screen

python scripts/benchmark_a100_roof_trace.py --allow-reassociation \
  --tunes -1 6 24 25 --samples 24 --rounds 4 \
  --output runs/o378_reduction_trace24

# 初筛后针对有收益候选测试四种模式，避免将无收益方案投入长测。
python scripts/benchmark_a100_roof_trace.py --allow-reassociation --all-modes \
  --tunes -1 6 24 --samples 24 --rounds 3 \
  --output runs/o378_reduction_four24
```

上述命令是验收流程，不表示结果已经通过；运行结果另行归档。

## 寄存器预算对照

首轮24/25未获得性能收益。O7同binary NCU显示：对照6、两路24、四路25的
local理论sector分别为10,485,760／31,064,064／82,575,360；MMA/I2F/FFMA数量各自不变。
这支持“增多的活跃累加器导致spill抵消收益”，但不是所有求和并行化均无效的证明。

后续26/27只将24/25的 `__launch_bounds__(256,2)` 放宽为 `(256,1)`，复用完全相同
的device计算体，独立编译以保留旧对照机器码。此举允许更多寄存器，可能降低spill，
也可能因只驻留一个CTA而损失延迟隐藏能力；不能预先宣称更快。
仍需显式 `--allow-reassociation`，不改变正式默认、5090或任何量化/scale语义。

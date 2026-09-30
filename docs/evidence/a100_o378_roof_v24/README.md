# v24：G128-major 低位输入布局候选

状态：已实现，等待 A100 实测；不修改正式默认或当前最佳结果。

## 假设与控制变量

此前最佳22/23的K128流水线中，NCU Source记录LDGSTS shared wavefront
16,777,216，其中excessive为8,388,608；LDSM excessive为0。
不能因此直接断言普通shared bank conflict是瓶颈。本轮测试全球内存中相邻
G128行片段的连续性，是否能减少搬运开销或改善供数。

- 41对应22的两阶段流水线；42对应23的三阶段流水线。
- CTA仍为64×128×128、128线程；共享内存排列、MMA fragment复用、
  独立G128 scale、group顺序及FP32 FMA顺序不变。
- 激活由`[2,M,G,64]`重排为`[2,G,M,64]`，权重由`[N,G,64]`
  重排为`[G,N,64]`，其中G=K/128，64表示packed bytes。
- 只移动原有字节，不重新量化；候选必须与旧实现输出逐位一致。

## 计时与验收

当前采用独立的16B向量重排kernel，尚未与转换融合。所有buffer提前分配。
重排成本计入conversion-only/cold：A额外读写`2*M*K`字节，W额外读写`N*K`字节。
steady-state缓存W，但仍计在线A重排；compute-only提前完成两者。
不把这些成本藏到免费离线预处理。

先执行源码/CPU布局索引测试、CUDA构建、同kernel原生INT4与异步搬运审计、
旧kernel代码生成一致性、逐位布局和输出验证、memcheck/synccheck/racecheck。
随后同binary换序配对测量22/23与41/42；保留CV失败，不挑最快轮次。
如初筛有收益，再做24真实样本和完整四模式；用NCU检验LDGSTS、共享wavefront、
指令数、寄存器与spill是否实际改善。暂不承诺布局优化一定有效。

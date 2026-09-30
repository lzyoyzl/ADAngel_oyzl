# v26：保留 B fragment 复用的 N64 候选（待 A100 验证）

目的：v24 的4-warp N128 GEMM实际使用168寄存器/线程，最多3 CTA/SM。
候选45/46将CTA改为`64×64×128`、WM2/WN2四warp，输出accumulator由64降为32/线程。
保持G128-major全局payload、B片段跨两个M atom复用、原有共享内存swizzle、
两路原生INT4、逐G128 scale和升序FP32 FMA；不改量化、不改求和、不使用magic-bias。

- 45/46分别使用2/3个`cp.async`stage，launch bound为128线程、4 CTA。
- 直接复用41/42的device body，独立TU只改变模板N和launch bound；旧实例保留。
- 不是重复旧9/10：后者为8 warp，WM4/WN2，没有当前跨M片段的B复用。
- 候选使用与41/42相同的**非融合**转换及payload重排，隔离GEMM实验。
  43/44转换融合不混入本轮，默认也不切换。

假设是减少寄存器和提高驻留CTA能改善延迟隐藏，**不是已经实现的性能收益**。
代价是N方向CTA翻倍，A载入重复次数、CTA同步/地址管理工作增加；即使零spill也可能更慢。

具体工作量预估：每个G128，原N128 CTA读A低/高共8KiB、W8KiB；新N64读A8KiB、
W4KiB，但CTA数量翻倍。因此全矩阵的payload搬运请求从16变成24个相同单位，增加50%。
这里是CTA请求量，缓存可能合并，**不等于DRAM实测增加50%**。数学MMA/I2F/FMA工作量不变。
目标驻留warp数从12到16（增加33%）也不等于性能增加33%；需要同时核对local访问、
shared wavefront、eligible warp与实测延迟，不能单凭occupancy判胜。

本地CUDA12.5独立TU编译通过：六个实例126–128寄存器、编译器报告零spill。
这不是A100 CUDA12.8结果，也不是GPU安全性或性能验收。132项CPU契约测试通过。

下一步待v25四模式测量结束后，同步GitHub源码至A100，构建并核对旧SASS；
原生INT4/cp.async审计、奇数G128与不同scale合成数值测试、memcheck/synccheck；
再执行同binary交错初筛。只有超过41/42才继续24样本与完整四模式，
所有MSE、CV失败和性能退步均保留并报告。

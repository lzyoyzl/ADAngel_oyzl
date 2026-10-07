# O7/O8 固定 high×16 的指令路由检查（v117）

状态：单一独立候选，等待编译成本检查；没有新增性能或MSE成绩，不改正式默认。

## 为什么不是重复优化

v71处理可变scale与partial的移位/加法，v87把可变激活因子改为CLZ＋系数移位，结果更慢。
本次只改变两路INT4合成中的固定high×16：量化、系数形成、整数累加顺序、八条MMA链、
32个partial/64个累加寄存器、CTA64×128×128、四warp/两stage和转换/epilogue均不变。
不重新测试tile、stage、producer、partial存储或累加链数。

最佳v78的整数循环383条指令，IMAD族158条（含地址运算），固定high重构混用IMAD.SHL与SHF。
最佳O8的预热NCU中FMA/ALU管线活跃比例约37.96%/21.65%，提示可能存在路由空间，
但这些百分比不是可相加的延迟，也不能据此推算加速。

## 精确变换

使用公开PTX的两源funnel shift：d=(b<<4)|(a>>28)。取b为high partial的原始32位，
a为threadIdx.x。实际CTA128线程，甚至SM80合法CTA最大1024线程时，a>>28恒为0。
G128 signed-high partial绝对值不超过8192，重构×16不溢出INT32；无新的舍入、定点误差或signed-shift UB。
非零下字只为避免编译器将简单左移重新选成IMAD.SHL；不修改SASS或CUDA/CUTLASS系统库。

[NVIDIA PTX 12.8：shf定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#logic-and-shift-instructions-shf)。
[NVIDIA开发者对ALU/FMA指令路由的说明](https://forums.developer.nvidia.com/t/pipeline-operator-forwarding-for-integer-instructions-in-cuda/299761/12)
只作为路由假设，不作为本kernel的SM80延迟模型。

## 预先固定的投入门槛

1. 新旧控制kernel机器码一致；同entry保留两路原生INT4，每G各32条MMA。
2. 从第二次signed MMA追踪到第一次unsigned MMA，64个high分量全部由SHF重构，排除地址移位误计。
3. 分配寄存器不超过168、整数热循环LDL/STL为0；LDSM16与异步copy10不变。
4. 总循环指令最多增加1%，IMAD族指令至少减少10%。这是成本筛选，不是预测性能。

门槛失败就停止，不扫描seed/opcode相邻变体。通过后先核对synthetic输出、guard/fallback，
再直接进行24样本三轮交错A/B、逐位输出/MSE与计时对照；不做小规模性能初筛。
只有实测收益成立才继续四种模式、GPU安全和必要NCU验收。O3暂不迁移该路由。

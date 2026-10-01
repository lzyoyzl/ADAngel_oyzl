# v33：有界 unsigned 寻址候选（尚未上机）

59/60仅将v31/55/56 device body的m/n/k/total_n参数从int改为uint32_t，
不改公共kernel签名、CTA64×128×128、4warp、两/三阶段、G128-major布局、
两路原生INT4、scale乘法及逐G128 FP32 FMA顺序。旧候选与正式默认保持不变。

假设：编译器能省去部分有符号除法/地址扩展指令，或改善地址寄存器生存期。
这只是待验证假设；没有用magic-bias，也没有将partial或跨组累加改为unsigned。
新候选不与v32的M32 tile合并，避免混淆变量。曾有固定4096尺寸候选无收益，
不能预先保证本轮类型变化会改善性能。

安全前提继承host入口检查：正尺寸、M64/N128/K128对齐、
`M*K,N*K,M*N<=INT32_MAX`、两维grid<=65535。最大payload/scale元素索引
在此范围内；FP32指针的字节缩放仍是原生指针运算，不截成32bit地址。
测试穷核有界表达式的极端形状、16B scale对齐，device源码差异只允许这几处类型变化。

本地CUDA12.5独立编译：59/60均168寄存器，stack为8/16字节。
这是编译预检，不是A100 CUDA12.8性能结果。需完成上机逐位/FP64参考、MSE、
同function INT4/async审计、有限内存同步安全、真实24样本配对和必要NCU。
筛选阶段prepared-core-only，四模式入口拒绝59/60，未获收益不接入正式结果。

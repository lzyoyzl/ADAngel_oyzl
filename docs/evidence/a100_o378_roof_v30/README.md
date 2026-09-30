# v30：行 scale 后移 + G128-major W scale（53/54，待实测）

v29的NCU仍有8,126,464个global excessive理论sectors；本轮检验W scale供数布局是否
能继续改善性能，不将该counter等同于实际HBM流量或直接预测加速比。

保持51/52的同一device body、64×128×128 CTA、4 warp、两/三stage、原生两路INT4、
G128独立scale和升序FP32累加，仍最后乘行A scale。新TU仅将GroupMajorScale设为true。
自然W scale输入仍是uint8[N,G]；在预分配buffer中重排为[G,N]，不修改scale值。

- prepared compute-only在计时前重排；full cold/conversion-only计入W转换，steady缓存。
- 4096³额外scale重排读写256KiB。A转换仍2个kernel，W变为3个，元数据不再声称两者相同。
- 不改变既有guard，不接受O7/O8，不回退其他数学路径。
- 53/54必须分别与51/52输出逐位相同；并继续验证相对原正式/O0的MSE和FP64语义参考。
- 不改正式默认或5090。只有完整实测后更新最佳报告。

流程：本地编译/CPU契约→GitHub→A100同步构建→旧代码保持/原生INT4审计→
数值和有限安全测试→4096配对初筛→24样本性能/MSE，必要时NCU与四模式验证。

本地预检：174项CPU测试通过；CUDA12.5独立TU编译通过。53为168寄存器、无stack/spill；
54为168寄存器、16B stack、12B spill store/load。这不是服务器CUDA12.8的资源验收结果。

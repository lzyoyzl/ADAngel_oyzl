# v31：O7/O8 G128 FP32 scale 异步搬运（待服务器验收）

候选55/56分别对照41/42，两/三阶段。保持64×128×128 CTA、4 warp、
168寄存器预算、B fragment跨M复用、G128-major payload和原生两路INT4。
仅将A/W的FP32 scale由标量LDG→寄存器→STS改为16B cp.async；
scale和payload使用同一个ring slot及commit/wait/CTA barrier。

scale本来就是FP32 group-major，不增加转换或重排。仍每G128计算
`scale=__fmul_rn(A_scale,W_scale)`并按升序执行FP32 FMA。
不采用树形求和、magic bias或跨G128整数累加；必须与正式版及41/42逐位一致。
仅接受O7/O8、16B对齐scale指针，prepared接口包含单组和奇数组G128检查。
正式默认、O3和5090实现不变。

旧14/15也探索过异步scale，但使用K256/256线程；本轮是在当前K128/128线程、
G128-major payload与B复用实现上单独验证，不能沿用旧结果或预先承诺收益。

本地CUDA12.5独立编译通过：55为168寄存器、32B stack及spill store/load；
56为168寄存器、16B stack及spill store/load。这不是A100 CUDA12.8的验收结果。

待执行：CPU契约→GitHub推送→A100同步构建→旧代码保持与原生INT4审计→
逐位正确性/有限内存同步检查→4096合成配对→24真实样本/MSE，必要时NCU及四模式。
所有离群和CV失败保留；未实测前不更新最佳性能数字。

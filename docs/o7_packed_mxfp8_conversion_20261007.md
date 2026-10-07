# O7：packed MXFP8 转 Q8（v126）

状态：独立候选，尚未 GPU 验收，不改正式默认或5090。GEMM目标仍未达到。

## 动机与查重

v125缩短MMA链反而增加整数工作；不重做链数、CTA、pipeline或shared因子缓存扫描。
另一个设想是把IMAD换成DP2A以转移执行管线，已取消：NVIDIA说明GA100的IMAD、IDP均使用FMA管线，不能声称换指令就能卸载该管线。[NVIDIA开发者说明](https://forums.developer.nvidia.com/t/separate-cuda-core-pipeline-for-fp16-and-fp32/302018/7)

本轮采用有正面先例的packed转换：v123在O8有效，但其E2M3/F=2解码不能直接用于O7的E4M3/F=−2。
不同于v34标量取整、v106逐元素warp查表、v118权重nibble布尔解码，不重测原方案。

## 唯一新机制

一个32位字并行处理4个E4M3编码，根据指数位选择精确的RNE除法或整数左移；以byte隔离的符号处理生成Q8，再用DP4A计算精确平方和。无需逐元素浮点转换、取整或warp查表。
所有254个合法编码、正负零、ties-to-even、byte carry、packing均核对。两个NaN编码仍由既有源格式契约拒绝；内部算术保留旧decoder的数值行为并不意味着接受NaN源数据。

对照使用已测v106激活转换和v118权重转换，候选只替换激活decoder。两边都保留v73行融合metadata/guard、同一v78原生双INT4 GEMM、量化、FP32输出及四种计时口径。

## 预设投入门槛

原LUT激活和packed权重entry编码不变；相对LUT转换至少减少5%静态指令；不超过32寄存器、无local/stack，shared仍256B、1次CTA barrier、4条DP4A，实际仍8CTA/SM。
门槛通过后才进行GPU数值和完整24样本×三轮四模式配对，warmup1000/repeats200/inner100；不做小规模性能初筛。失败即停止，不扫描相近Boolean/LUT参数。

编译指令减少不是性能结果；只有直接Event测试可确认conversion、Cold和steady收益，也不能把转换收益宣称为GEMM吞吐提升。

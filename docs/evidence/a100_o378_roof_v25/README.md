# v25：转换直接写入 G128-major payload（等待 A100 验收）

候选43/44分别复用候选41/42的**同一GEMM函数**，不是新MMA/CTA实验。
SM80独立转换TU将源格式直接解码/定点化到 `[plane,G128,row,64 bytes]`，
O7/O8同时写出原有group-major FP32 scale；O3保留原有row-major UE8M0 scale。
不改变量化、RNE、符号、scale乘法或G128 FP32累加顺序。

四种计时方法与前版相同；原独立重排消失后，不再收取额外A32MiB/W16MiB读写。
源格式读取和最终payload/scale写入仍完整计费。验证用自然布局导出在所有计时结束后
由物理输出逆排列生成，正式GEMM不依赖这些副本。所有实际计算buffer预分配。
正式默认、所有SM120文件不变。

验收：有限码值/微指数/scale/尾块转换逐位检查；三种shape和四种计时模式；
独立语义参考、旧41/42逐位对照、内存/同步检查；对照GEMM机器码审计；
24样本配对Conversion/Compute/Cold/Steady、MSE和所有CV记录。

结果尚未生成，不声称性能收益。即使转换更快，也必须直接测量端到端，不能相加median。

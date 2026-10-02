# v66：O8 FP6无分支转换初筛，负收益，不采纳

本轮只测试一个**数值完全等价的activation conversion候选**。
O7/O8全K整数GEMM仍待确认，没有越过该确认范围；O3/O7/O8 GEMM、量化规则及正式默认均不改。

## 动机与改动

v53已归档NCU显示，FP6向量16转换执行约1.049M条BRA；选择只去掉其payload译码分支。
保留16元素/线程、G128-major输出、packing及有效scale的FP32乘法顺序。
令`x=code&31`：低两档指数的定点幅值为`RNE(x/2)`，高两档为`(8+(x&7)) << ((x>>3)&1)`。
显式`selp`选择幅值，再以符号掩码恢复正负。穷举全部64种E2M3编码与原RNE结果一致。
只是局部等价改写，不更改F=2、不改G128、不改变O8输出舍入顺序。

## 编译与审计

|FP6 vector16指标|原版|无分支候选|
|---|---:|---:|
|寄存器/线程|19|31|
|静态SASS指令|488|416|
|静态BRA|33|1|
|stack/local bytes|0/0|0/0|
|LDL/STL|无|无|

剩余分支不应称为payload译码分支；没有声称整个kernel没有任何分支。
20个非目标转换entry编码SASS完全相同；控制的8个vector entry与v53旧最佳编码完全相同。
GEMM未重编译，正式扩展SHA256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

首次审计误将非目标entry数量设为18，漏计两个同TU的O3转换entry，因此报告FAIL。
核对全部symbol后改为20并重新审计，**没有重新编译或改变GPU二进制**。
原`audit.json`保留，复核结果为`audit_rechecked.json`，不是忽略指令差异。

## 四真实样本配对结果

A100，首层q/k/v/o共4样本，4096² activation，3轮，warmup50/repeats200/inner100。
每个Event区间重复100次除以100；单stream、预分配、按样本/轮次交换执行顺序。
共享未锁频GPU，24条记录全部保留，CV全部小于3%。

|计时对象|原版 ms|候选 ms|配对吞吐变化|配对speedup 95% CI|
|---|---:|---:|---:|---|
|O8-A FP6→Q6/INT8并拆两路INT4|0.04020736|0.04176896|**−3.85%**|[0.951546, 0.980750]|

ms先在各样本跨轮取median、再跨样本取median；speedup先做同输入配对，因此不等于两列ms直接相除。
这是**单operand转换时间**，不是conversion-total、GEMM-only、Cold或steady-state。
四样本均未显示收益；停止候选，不扩大24样本、不扫描别的无分支表达式。

## 正确性与MSE

GPU转换验证393项通过，18项非法输入拒绝；覆盖有限编码、scale编码、尾行、奇数组与非默认stream。
4样本×3轮×2实现的packed payload与scale均与Python参考逐位一致。
把转换结果送入原O8/GEMM59，8项输出检查均finite FP32且与旧最佳逐位一致。

|指标，4样本|原版与候选|
|---|---:|
|O8相对O6输出MSE median|0.000123493256717198|
|O8相对O6输出MSE mean|0.000122317650852208|
|新旧O8输出差MSE|0|

本表不是24样本精度结果，不能据其较小值声称精度提高。
本轮没有新增NCU或sanitizer；不把语义验证称为内存安全专项验收。

## 为什么停止

静态分支和指令数减少没有转化为性能收益。寄存器增加，且无分支表达式始终计算两个候选幅值，
原分支可能跳过部分算术；这些是可能机制，不是本轮NCU已分离的归因。
因此不能仅凭“无分支”推断更快。已有稳定负收益，按预算约束不继续花费profiling/全量测试。
保留旧向量转换5，完整最佳与正式默认不变；不把这一转换实验计作GEMM优化。

## 复现与身份

编译提交`5672731e8bef89301b27151ed6c0ca3603cbf8b3`；复核/测试提交
`58cc8b35fe7a2a374fe446673112dfab20f58747`。均先本地实现并push，再由A100 fetch/ff-only同步。

```bash
python scripts/probe_nv6_branchless_codegen.py --output reports/o378_roof_v66_rebuild
python scripts/validate_vector_conversion_probe.py \
  --library reports/o378_roof_v66_rebuild/policy_1/libconversion.so \
  --output reports/o378_roof_v66_revalidate
python scripts/benchmark_nv6_branchless_probe.py \
  --directory reports/o378_roof_v66_rebuild \
  --output reports/o378_roof_v66_rescreen --samples 4 --rounds 3
python -m unittest discover -s tests/unit -p test_nv6_branchless_codegen.py -v
```

新脚本构建时已使用正确entry数量，不需要对新构建执行`--reaudit`。
本地归档保存原始Event、MSE、验证、生成源码、SASS和资源日志；`.so`留在A100 build目录。
控制库SHA256：`07d1ae71085c759ba581c4023029faeef3c24047db7437010ea0f4ea7e7bcf4c`。
候选库SHA256：`c1226b92bd5e5b00abef4ea6c606a20b416f07edf611fa932f18f4d32d66d10a`。
传输包`tmp/o378_roof_v66_evidence.tgz` SHA256：
`f1db5715594ff2decc058808ebba4d49ef3b5d67ae463820e51f2a30f0a01f8a`。

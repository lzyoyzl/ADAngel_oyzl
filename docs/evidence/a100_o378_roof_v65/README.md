# v65：O7/O8 真实 payload 的整数范围检查

这是**只读可行性检查，不是新的 GEMM 性能迭代**。源量化、正式后端、默认及5090实现均未修改。
先前v56用每个元素的最大可能值估计范围，较为保守；本轮使用同一24样本的实际定点整数，
检查是否有希望把逐组FP32后处理改为精确整数合并，再恢复最终实数scale。

## 结果

A100，原始FP16 trace重新调用现有源格式量化和定点参考；每样本4096³、32个G128。
先检查首层4样本，再独立检查全部24样本；前4样本的数据与结果完全一致。

|路径|整张输出通过安全界|未通过安全界的输出坐标|未通过的64×128 CTA|CTA通过比例|
|---|---:|---:|---:|---:|
|O7|24/24|0/402,653,184|0/49,152|100%|
|O8|23/24|15/402,653,184|12/49,152|99.9755859375%|

O8的所有例外均在`layer_24_o_proj`。**未通过充分安全界，不等于实际已经溢出**；
不能把这些坐标直接放行。CTA粒度保守回退意味着相应12个CTA都使用原计算，而非仅15个输出。
两侧整数factor相乘的额外检查在48个case全部通过，最大保守乘积O7为294,912、O8为917,504。

## 安全界如何得到

使用现有定点整数`q_A/q_W`，不更改量化、不重新截断payload。
把源scale精确解码为`odd_mantissa × 2^exponent`；每行/列选择其所有非零scale的最小指数，
得到非负整数factor：`scale[g] = factor[g] × 2^anchor`。

公共实数因子分别为O7的`4*T_W`和O8的`T_A/4`。
这只是在**实数代数**上提取公共因子，不能声称保留现有FP32中间舍入。
HiF4内部微指数已经包含在现有`to_fixed_reference`生成的整数payload内。

对每行/列计算：

```text
A_norm2[m] = sum_g,k(q_A[m,g,k]^2 * A_factor[m,g]^2)
W_norm2[n] = sum_g,k(q_W[n,g,k]^2 * W_factor[n,g]^2)
```

由Cauchy不等式：

```text
sum_g(abs(P[m,n,g]) * A_factor[m,g] * W_factor[n,g])
    <= sqrt(A_norm2[m] * W_norm2[n])
```

因此`A_norm2[m] * W_norm2[n] <= (2^31-1)^2`是每个组乘积与**任意前缀整数和**都安全的充分条件，
不只是最终输出安全。实际判断使用Python任意精度整数比较，不计算浮点sqrt，不截断大指数。
另检查`max(A_factor) * max(W_factor) <= 2^31-1`，避免系数相乘先溢出。
CTA判定使用该tile的最大行/列norm和factor，保证分支可保持CTA一致。

GPU仅用INT64计算每G128的未加权整数平方和；解码、加权及范围判断在CPU进行。
全24样本检查耗时35.1887秒，包含源量化及数据处理，**不是在线GPU guard成本，也不是GEMM延迟**。

## 结论与下一步门槛

- 新证据表明：此前仅由scale范围得到的“不适合整矩阵直接放行”，不应被理解为真实payload普遍溢出。
- O7值得考虑独立整数合并候选；O8至少需要安全检查及原算法回退。
- 激活相关的安全检查必须在线付费；静态权重准备可按版本缓存。不能让CPU检查免费出现在GEMM计时外后宣称端到端收益。
- 源量化和各G128 scale必须保留；FP32舍入顺序变化需用FP64语义参考、24样本MSE和finite检查重新验收。
- 尚未实现新的O7/O8整数GEMM；没有新ISA、sanitizer、MSE或四模式性能结果，不能声称获得加速。
- 现有完整最佳组合、O3独立v62结果和正式默认均不变。O7/O8扩大整数累加范围的候选等待用户确认。

## 补充：在线guard与指令工作预算（仍未实现候选）

从已归档数据重新分析，无新GPU运行。对每行/列取严格向上整数平方根
`R=ceil(sqrt(norm2))`，CTA用`max(R_A)*max(R_W)<=INT32_MAX`及factor乘积界判断。
该判定比原平方和乘积界只会更保守；48个case实测**没有新增回退CTA**，仍只有O8的12个。
原平方和最大37位，向上平方根最大367,307；当前数据可用UINT64保存norm及根的乘积。
这不是对任意新输入的范围保证，实际GPU准备仍须检测溢出/非法编码并回退或拒绝。
普通浮点`sqrt`后转整数不自动满足向上界，不能直接代替此处的精确整数证明。

|假设的准备方式|新增逻辑工作|计时要求|
|---|---|---|
|另开activation检查pass|至少额外读取16 MiB定点activation payload，再准备factor/norm等|计入在线转换、Cold和steady，不是免费检查|
|与现有activation转换融合|避免再读payload；若输出逐G128 UINT64平方和，则写1 MiB、后续归约再读1 MiB|仍需跨G128行归约、CTA判定及新增寄存器工作|
|静态W准备|factor面板512 KiB、行元数据64 KiB/operand的拟议布局|Cold付费，按权重版本缓存；不能跨修改复用旧证明|

上述是选定布局的逻辑字节数，**不是实测DRAM流量或延迟**。A的factor面板同样为512 KiB，
2048个CTA的4字节flags为8 KiB；保留原scale用于回退时须同时保留相应旧buffer。
现有向量转换每线程16元素，CTA覆盖同一G128的32行；它不能在没有跨CTA归约的情况下
直接得到该行全部32组的norm。因而“融合转换”不等于自动零成本完成guard。

已有O7/O8 tune59单样本NCU各为116,285,440条动态warp指令；逐组I2F/FMUL/FFMA各16,777,216条。
在假设每次“转换+scale相乘+FMA”三条工作变成“整数factor相乘+整数MAD”两条的简化模型中，
可减少16,777,216条，约**14.43%**；尚未扣除最终524,288条I2F、epilogue scale和其他新增工作。
这是源码工作预算，**不是候选SASS、严格性能上界或预计提速百分比**。
两路IMMA仍为16,777,216条，MMA必要容量下界并未降低。
O3已有全K实测也表明整数处理与spill可能抵消I2F减少，不能据此承诺O7/O8接近0.220347ms。

因此若获准，先做一个编译/小样本门槛：检查实际整数指令、寄存器、spill与准备成本，
有明确配对收益再扩大24样本；不提前投入多版guard或tile扫描。
[逐样本guard与工作预算JSON](guard_cost.json)由下列命令生成，新增3项CPU回归逐项复算：

```bash
python scripts/analyze_o78_integer_guard_cost.py --output reports/o78_guard_cost_check.json
python -m unittest discover -s tests/unit -p test_o78_integer_guard_cost.py -v
```

## 证据与复现

采集脚本提交：`44879ebb34bdb56dc448a3f26ccd2774987b1f3c`。
采集脚本SHA256：`933f244485bf25a8458fcd4b4b820d80d9658486c591a91c5c691dedde2875ff`。
原始trace manifest：`4ff05585d91f8940f20328b140c637ac948d0dbe47f1506db0cb5d839c9c3db0`。
prepared manifest：`05849422f6d8ad18e7c3468ff6af549d33ce86eeb9590af7388ab3452d26881c`。

```bash
python scripts/inspect_o78_payload_integer_bounds.py --output reports/o378_roof_v65_screen --samples 4
python scripts/inspect_o78_payload_integer_bounds.py --output reports/o378_roof_v65_trace24 --samples 24
python -m unittest discover -s tests/unit -p test_o78_payload_integer_bounds.py -v
```

输出要求新目录。7项CPU测试穷举有效scale编码，核对安全阈值、随机前缀界、导出hash与篡改拒绝，
逐文件重算全部56个case的输出/CTA计数，并从例外case完整分组数据重算norm。
本目录保留所有样本的行/列norm、anchor、factor上界及payload/scale身份，例外case保留全部group统计。
通过的case省去重复的分组统计，原文件hash仍保留；完整数据留在A100上述目录及本地传输包。

完整传输包`tmp/o378_roof_v65_evidence.tgz` SHA256：
`f7dcc643c10599f94e6cd3239657fcae65857331aa4e422002c5e6618333a469`。
原4样本summary SHA256：`e3ca6f92c3257ab97d119b9d6cca6fff00928e4563ef15c69b46abb81a9d4557`。
原24样本summary SHA256：`5ab1a857b2d33c167826b283ea076913ba0b9802642811254456ce65ee898c3d`。
导出manifest同时记录原summary、原case及导出case的SHA256；导出不是重新测量。

# v136：当前最佳 O3，预热后完整 NCU

采集代码commit=`4e5c357ae6d164feb8c98518d614e1fb4ea1c77b`。
当前v89原GEMM、两路原生INT4、原输入与量化语义；正式默认、扩展、5090不变。

- `layer_12_o_proj`，4096³，全部2048CTA走整数guard路径。
- NCU full / application replay / cache none / clock none；每次1000次预热、1次目标launch，共50份成功receipt。
- 每次均复核选定raw/prepared SHA、实际加载cubin、扩展与输出；full24数据哈希在采集前验证。
- 输出均为finite FP32，相对原v79全K实现逐位一致；vs O0 MSE=`0.0003752320504872409`。
- 原GEMM cubin SHA=`89918880f8cad38f8234993ff07064d5c8bb6f2c5a4996c3d443d637836856bd`。
- 正式扩展SHA=`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，前后相同。

`reports/o378_roof_v136_o3_warm_ncu/`包含原始raw/source CSV、50份receipt、输入/构建与命令记录、
NCU日志、版本和analysis；旁边保存父进程完成日志。`index.json`固定66份原始文本SHA。
`phase_analysis.json`为逐PC消费者角色，核对2200条解码指令/323条热循环与全部动态计数。
原v133 O7/O8分析器及证据未改写；本轮不使用O8百分比代替O3。

完整归档含 `.ncu-rep`、适配器SO和conversion cubin，保存在本地与A100：
`tmp/o378_v136_o3_warm_complete.tar.gz`。
SHA256=`36df8d4afc99dc2980e4bbb7dfe18d7697dbe6f8c379d629d6f023c835d5044b`。
这些二进制不提交Git。A100原报告：
`reports/o378_roof_v136_o3_warm_ncu/o3_warm.ncu-rep`。

本轮没有新优化候选、新24样本Event结果、加速或sanitizer结论。
NCU的0.376160ms不能与跨轮Event中位数相除作为提速。

[诊断结论](../../o3_best_warm_profile_20261007.md)。

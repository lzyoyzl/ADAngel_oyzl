# v133：既有 NCU 的消费者阶段分析

不是新kernel、编译、GPU运行、NCU采集、性能或MSE结果。当前最佳、正式扩展和5090不变。

`analysis.json`由 `scripts/analyze_o78_mma_phase_stalls.py` 生成，包含分析脚本SHA、4份既有输入SHA、383条热循环指令角色、分类动态指令/未发射采样计数、30个热点PC及条件CALL有效执行计数。
原始输入直接引用仓库中的v78/v114证据，不复制或改写旧记录。

核对全入口1976条解码指令的操作数/predicate；允许NCU显示的分支地址重定位、reuse省略及标点差异。
分类总数闭合到105.521152M动态warp指令、10557个未发射采样。**不是二进制编码等同性审计，也不是因果耗时分解。**
MMA占wait样本50.85%；两步整数scale占22.76%。这些是同一旧O8样本的进一步归类，不是新的跨24样本结论。

```bash
python -m pytest tests/unit/test_o78_mma_phase_stalls.py -q
```

完整解释和查重：[消费者阶段瓶颈报告](../../o78_consumer_phase_bottleneck_20261007.md)。

# v90：当前 O3 / O7 的配对 NCU 证据

这是已验证候选的瓶颈诊断，不是新一轮 CUDA Event 性能测量。
原始 GPU capture 均在 A100 执行，分析脚本修复后仅重算 CSV，没有重跑 GPU。

|文件前缀|实际 kernel|版本|
|---|---|---|
|`o3_0`|`adangel_roof_o3_eight_chain_candidate`|v79 对照|
|`o3_1`|`adangel_roof_o3_grouped_cta_candidate`|v89 候选|
|`o7_0`|`adangel_roof_o78_eight_chain_candidate`|v78 对照|
|`o7_1`|`adangel_roof_o78_grouped_cta_candidate`|v89 候选|

每个 capture 为 `layer_00_q_proj`、4096³、50 次不计时预热、1 个正式 kernel；
`--set full --cache-control all --clock-control none`，每份报告重放 50 passes。
单 CUDA stream，2048 个整数路径 CTA，回退/非法 CTA 均为 0。
这是四份单样本诊断；没有重新采集 O8，没有产生新 conversion/Cold/steady 数据。

目录 `reports/o378_roof_v90_ncu/` 保留 raw CSV、source/SASS CSV、分析 JSON、
命令、日志和构建/输出身份收据；同 entry 静态 SASS 指纹、原生两路 INT4、
动态 MMA 工作量、有限 FP32 和相对旧最佳逐位一致全部通过。
O3/O7 的该样本 MSE 见各 receipt，不能替代 24 样本 MSE。

完整二进制报告和驱动保留在 A100 原目录及本地归档，不提交 Git：
`tmp/o378_v90_ncu_complete.tar.gz`，两端 SHA256：
`78e0b0e7d3f3d35e8bcef20d02a3f038018e4713b9783f3000c31c6b85698b1e`。

复核（不重新采集 GPU）：

```bash
python scripts/run_grouped_cta_ncu.py --variants o3 o7 \
  --output docs/evidence/a100_o378_roof_v90_ncu/reports/o378_roof_v90_ncu \
  --analyze-only
python -m pytest tests/unit/test_roof_v90_evidence.py -q
```

首次离线解析拒绝 grouped kernel 名称，原因是分析器的许可列表仍只有旧 kernel。
修复只新增明确的 variant/tune/symbol 组合，保留所有工作量及指纹验证；
没有把 renamed parser fixture 当作 GPU capture，也没有放松为任意 symbol。
O3 profile 的 `build/*.json` 身份收据需显式纳入 Git，不能被通用 `build/` ignore 掩盖。
正式默认和 RTX 5090 实现不变。

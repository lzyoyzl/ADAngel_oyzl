# v114：当前最佳 O8 的预热后完整 NCU 证据

这里只补齐尚未采集的诊断，不是新 kernel 或加速实验。O8 使用原 v78 GEMM CUBIN、
v73 准备库；正式扩展、默认、O3/O7、trace、转换和 RTX 5090 均未改动。

- 真实 `layer_12_o_proj`，4096³；全部2048 CTA走安全整数路径。
- NCU 2025.1.1，`--set full --replay-mode application --cache-control none --clock-control none`。
- 50个应用回放进程，各自50次预热后只采一个正式入口；50份完整 receipt 全部保留。
- FP16原始样本和prepared数据预检覆盖24样本；每进程复核选定文件与v99源格式identity。
- finite FP32、相对原v67全K输出逐位相同，输出差MSE=0；本样本 O8/O6 MSE=`0.00018277407059992296`。
- 真实入口源SASS指纹、CUBIN SHA、原生U4/S4与S4/S4、动态数学工作量及资源检查通过。

`reports/o378_roof_v114_o8_warm_ncu/`保留原始raw/source CSV、50份receipt、
validation、commands、NCU版本、capture日志和可重算analysis；旁边为父进程完成日志。
`index.json`固定58份原始文本的SHA及完整归档SHA。不是重新运行24样本性能验收或sanitizer。

完整归档（含 `.ncu-rep`）在本地 `tmp/o378_v114_o8_warm_complete.tar.gz`，
SHA256=`87ce0802875333908270013e6dd21f3b78c2c34f6ad10179d89e4f81f5c2e614`。
二进制保留在A100原reports路径和本地 `tmp/o378_v114_o8_warm_extracted/reports/o378_roof_v114_o8_warm_ncu/o8_warm.ncu-rep`，不提交Git。
采集源码commit=`e9c6556595875993bd1251e6f96084c2b84e7590`。

```bash
python -m pytest tests/unit/test_o8_best_warm_ncu.py tests/unit/test_roof_v114_evidence.py -q
# 仅重新分析已有报告，不启动GPU或再次采集：
python scripts/run_o8_best_warm_ncu.py --output reports/o378_roof_v114_o8_warm_ncu --analyze-only
```

解释及查重见 [预热后诊断报告](../../o8_best_warm_profile_20261007.md)。

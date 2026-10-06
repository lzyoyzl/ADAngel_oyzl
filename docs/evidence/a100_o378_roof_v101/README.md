# v101：当前最佳 kernel 的 CTA/warp 时间线诊断

不是新优化版本，不改变正式默认、扩展、量化或5090；不是24样本性能验收。
采集 source commit：`dd5a9cba46b5b857dd021bbc4211112fb372f3c5`；
编译 source commit：`ed51f83b7120fd4d762fc5ced4b2f59b42b67935`。

`reports/analysis.json` 是从9份 `*_timeline_*.npy` 重算的规范汇总。
各 `receipt.json` 保留四warp原始时间戳解析、每SM区间、400个控制Event样本、
3次 capture Event、GPU快照、单样本MSE及文件/代码/正式扩展身份。
codegen目录保留 SASS/PTX/liveness/resources、旧完整编码对照与生成源码。

完整原始归档 `tmp/o378_v101_complete.tar.gz` 在本地及A100项目内保留，SHA-256：
`2b3f12c70372571efad5f5b4549127807477af68b97cb1a249d6acefa213bc17`。
归档含 cubin/driver `.so` 与首次 guard 检查失败日志；二进制编译产物不提交Git。
保留Numpy时间戳作为可重算的原始数据，不把它们误当模型或partial payload。

重算（输出必须不存在）：

```bash
python scripts/summarize_cta_timeline.py \
  --evidence docs/evidence/a100_o378_roof_v101/reports \
  --output reports/o378_v101_recalculated.json
python -m pytest tests/unit/test_roof_v101_evidence.py -q
```

时间戳刻画warp观察点，不是精确CTA资源分配/释放；插桩可能扰动代码，
O7/O8整个entry还新增8B local，热循环local仍0。缺失槽位折算约5%不是可获得加速。
未锁频、原始CV失败全部保留，不混入正式最佳结果。完整说明见
[诊断与查重](../../o3_o7_o8_cta_timeline_20261007.md)。

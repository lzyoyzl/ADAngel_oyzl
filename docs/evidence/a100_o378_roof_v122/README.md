# v122：INT16 partial 跨 warp 交接，全24样本负向结果

已执行：全24范围证明、原生INT4/SASS审计、实际occupancy查询、数值/回退检查、
memcheck/synccheck/racecheck、24样本三轮交错配对 GEMM。候选正确但更慢，不采纳。

| Case | 原最佳 ms | 候选 ms | 配对吞吐变化 | MSE |
|---|---:|---:|---:|---|
| O7 | 0.478208 | 0.595968 | −20.07% | 与旧最佳完全相同 |
| O8 | 0.480256 | 0.601088 | −20.24% | 与旧最佳完全相同 |

288条记录，三轮，每轮warmup1000/repeats200。所有原始Event数据保留；仅compute-only，
CPU新范围判断在计时外缓存，**没有新增conversion/Cold/steady-state测量**。

`index.json` 固定46份原始文本的字节数和SHA。编译源码 `499b6efa8c1f79b52819790af94f9593090a713c`，
运行源码 `d6bebfe0887c8aa78a130659c207cbab51ec1e44`；均先推GitHub，再由A100项目fetch/ff-only。
原始二进制和48份平方和NPZ未上传Git，保留在本地与A100：

```text
tmp/o378_v122_complete.tar.gz
SHA256 80fdf12a77101fbf6b048e21814b51c2b5a4c4882f354992ca788602e83473cd
```

正式扩展SHA仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
正式默认、O3、5090未修改。本轮没有NCU；不能将编译指令数量直接等同于时间。

`tmp/o378_v122_data.log` 是首次JSON序列化错误日志；修正后全24结果在 `*_handoff_data_r2`。
`tmp/o378_v122_memcheck.log` 是首次sanitizer筛选参数错误，未执行GPU；真正通过的memcheck日志为 `*_memcheck_r2.log`。
以上错误均保留，不能将其混同为数值/内存测试失败。

CPU复算原始计时、编译证据、源数据identity、MSE和归档完整性：

```bash
python -m pytest tests/unit/test_partial_handoff.py tests/unit/test_roof_v122_evidence.py -q
```

在已配置的A100项目复现（输出目录必须不存在；候选从未成为默认）：

```bash
python scripts/inspect_partial_handoff.py --output reports/handoff_data_recheck
python scripts/probe_partial_handoff_codegen.py \
  --data-gate reports/handoff_data_recheck --output reports/handoff_codegen_recheck
python scripts/benchmark_partial_handoff.py --codegen reports/handoff_codegen_recheck \
  --validate-only --output reports/handoff_validation_recheck

# 验证及同entry安全检查通过后，直接全24三轮，不做小规模性能筛选。
python scripts/benchmark_partial_handoff.py --codegen reports/handoff_codegen_recheck \
  --output runs/handoff_full24_recheck
```

[改动、结果、瓶颈及平台迁移说明](../../o7_o8_partial_handoff_20261007.md)。

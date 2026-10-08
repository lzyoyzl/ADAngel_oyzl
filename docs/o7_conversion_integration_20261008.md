# O7 最佳转换组合验收（v139）

本轮补齐此前的组合验证缺口，不是新的 GEMM 优化，也不把旧收益相乘作为新成绩。
当前 GEMM 最佳仍为 O3 v89、O7/O8 v78；正式默认和5090保持不变。

| 路径 | 权重转换 | 激活转换 | GEMM |
|---|---|---|---|
| 已测 v118 对照 | packed NVFP4→Q4 | 原 v73 E4M3→Q8 | 原 v78 |
| 本轮组合 | 同一 packed NVFP4→Q4 | 已测 v106 warp lookup E4M3→Q8 | **同一个 v78 CUfunction** |

v106/v118此前分别测过，本轮只验证它们组合后能否进一步改善转换和端到端。
复用v126冻结库的**control**分支；未通过编译门槛的v126 packed-MX8候选不执行，旧门槛不改写。
不新增CUDA编译、不重新安装依赖、不重编译正式扩展。

先校验旧库/源文件SHA、两条转换及guard完整SASS，再做32项合成四模式、边界拒绝、
非默认stream与逐位输出/metadata/norm检查；已测decoder本身未改动。
随后24样本×3轮四模式，warmup1000/repeats200/inner100，交错单stream，保留全部CV和原始Event。
直接比较v118，避免只与更慢的原始转换比较而宣称“又优化了最佳”。

```bash
python scripts/benchmark_o7_conversion_combo.py --validate-only --output runs/o378_v139_preflight
python scripts/benchmark_o7_conversion_combo.py --output runs/o378_v139_full24
python scripts/analyze_o7_conversion_combo.py --input runs/o378_v139_full24 --output reports/o378_v139_analysis.json
```

## GEMM 查重与范围

已复查v89/v78源代码、当前预热NCU和v92–138台账。当前整数热循环不是旧FP32 scale链。
更多链/warp、固定stage展开、fragment预排/预取、系数表及high/low分路已有编译或完整配对负结果。
本轮没有找到同时保持两路原生INT4、原scale且有新证据支持的GEMM改动，因此不重复这些试验。
约0.220347ms仍是1410MHz理想重叠的容量必要下界，不是本轮已经达到的延迟。

结果待A100完整验收；不得将此计划描述成已获得提升。

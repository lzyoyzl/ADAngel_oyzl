# A100 v93：固定环形槽位的编译筛选证据

源码 commit：`dce0b838a29811fbe99e80e6ce9ea5981e6a710a`。
两个候选完成 pinned CUDA12.8 / CUTLASS 的 cubin/PTX 编译、同 entry 原生 INT4/copy 审计、
编码对照与静态存活分析，预设 gate 均为 false。未执行 GPU benchmark、MSE、NCU 或 sanitizer。
本地/A100各102项相关回归通过，不将CPU/编译检查冒充GPU数值验收。

本目录只保存不可变文本，包含 SASS/PTX、liveness、完整编译日志和 JSON receipt。
二进制及全部文本另外存于本地和 A100 的 `tmp/o378_v93_compile_complete.tar.gz`：

```text
SHA256=6fe29ce67e8381360ac718918d233370e84f2ef8184d2937c3dd2e1d915a5068
```

公共 `_sm80` 扩展 SHA256 未变：

```text
94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462
```

复现使用不存在的输出目录，不覆盖原始证据：

```bash
python scripts/probe_stage_cycle_codegen.py --kind o3 --output reports/new_v93_o3
python scripts/probe_stage_cycle_codegen.py --kind o78 --output reports/new_v93_o78
python -m pytest tests/unit/test_stage_cycle_probe.py tests/unit/test_roof_v93_evidence.py -q
```

筛选会验证源码/二进制身份、同 entry PTX/SASS 和对照编码；若 gate 为 false，停止运行测试。
计数必须按每圈的2/3个G128归一化，不能把展开后的总数直接当作每组工作。
完整GPU正确性/性能/MSE结果仍引用此前最佳，并非本次候选结果。

[简明结论](../../o3_o7_o8_stage_cycle_20261006.md)。

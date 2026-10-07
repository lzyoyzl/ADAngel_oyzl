# v128 原始编译证据

源码先推送 GitHub，再经增量 bundle 在 A100 项目 fetch/ff-only 到 `67eff20593662f4c98b1d4ab549921ba67123c9d`。
CUDA 12.8.93 / pinned CUTLASS；原 v89 对照完整 SASS 编码不变。
候选 8 次下一组 A 加载之后有 32 次本组加权，但静态热循环 323→374，超过预设 +5% 门槛。
停止，没有候选 GPU 执行、数值/MSE、Event、sanitizer、NCU 或跨平台结果；不是实测慢 15.8%。

15 份原始文本原样保留，字节数/SHA 见 [index.json](index.json)。
完整压缩包 `tmp/o378_v128_verified.tar.gz` 包含 CUBIN；本目录只保留文本，CUBIN SHA 在 codegen.json。
压缩包 SHA：`8eb2479cd50602d38422c05cfa45e3729595c7a7c4d297c98eabc89a030f3021`。
正式扩展 SHA 未变，见 index；不把 CPU/source 测试当 GPU 正确性验收。

复现编译（仓库相对路径；输出必须是新目录）：

```bash
python scripts/probe_o3_crossk_load_codegen.py --output reports/o378_roof_v128_replay
python -m pytest tests/unit/test_o3_crossk_load.py tests/unit/test_roof_v128_evidence.py -q
```

生成器成功保存审计时可退出 0，是否值得执行 GPU 必须读取 `codegen.json.cost_gate.passed`，不能只看进程退出码。
不放宽原门槛、不扫描相邻调度、不迁移这个未通过候选到 O7/O8。

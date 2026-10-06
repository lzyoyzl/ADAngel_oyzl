# A100 v92：16 链 MMA 候选完整证据

同 CTA 的一个固定调度候选，非 tile/stage/寄存器上限扫描；正式默认不变。
首测与复测各 24 样本×3轮，共864条 Event记录，输出逐位一致、MSE不变，两次均无收益。
CV仍大量超标，检测到其他CUDA context，原始数据全部保留，不冒充严格稳定验收。

本目录只保留文本（SASS/PTX、编译/存活分析、原始计时、验证、环境、快照、汇总）。
二进制和全部运行 build 存放在本地/A100的 `tmp/o378_v92_complete.tar.gz`：

```text
SHA256=bae166987f8515f94f9c6e919f912bfe9a373040651e30a81a865b8c6ab0dc81
```

实际实现、编译与首测/复测的源码 commit 均为 `15aca3c454a341d2344fead7b7dad525ebd88b48`。
之后的统计校验工具提交为 `30f17279bd37f546d8e9d54c6b5051572a95a469`，不改变候选或对照 binary。
公共 `_sm80` 扩展仍为：

```text
94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462
```

## 复现

下列输出目录须不存在；使用A100既有环境、pinned CUDA12.8/CUTLASS及原始/prepared24样本。
先审计编译 gate；失败则停止，不运行小样本性能初筛。

```bash
python scripts/probe_sixteen_chain_codegen.py --kind o3 --output reports/new_v92_o3
python scripts/probe_sixteen_chain_codegen.py --kind o78 --output reports/new_v92_o78
python scripts/benchmark_o3_sixteen_chain.py --codegen reports/new_v92_o3 \
  --output runs/new_v92_o3 --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python scripts/benchmark_o78_sixteen_chain.py --cubins reports/new_v92_o78 \
  --output runs/new_v92_o78 --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python scripts/analyze_sixteen_chain_evidence.py --o3 runs/new_v92_o3 \
  --o78 runs/new_v92_o78 --output reports/new_v92_analysis.json
python -m pytest tests/unit/test_sixteen_chain_probe.py tests/unit/test_roof_v92_evidence.py -q
```

`reports/o378_roof_v92_analysis.json` 从每条原始计时重算 mean/median/CV、完整配对与MSE一致性；
同时保留首测/复测及同输入/kernel身份校验。没有新的NCU或sanitizer成绩，没有新的conversion或端到端成绩。

[本轮简明结论](../../o3_o7_o8_sixteen_chain_20261006.md)。

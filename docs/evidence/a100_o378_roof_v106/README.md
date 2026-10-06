# v106：O7 warp 分布式精确 MXFP8→Q8 转换

直接24样本×三轮×四模式×两policy，576条记录、345600个阶段Event值。
原始数据来自A100；源格式与v99完整SHA/identity一致，原始FP16直接量化。
转换batch摊销，GEMM/Cold/steady单次直接计时；无离群过滤、无最快轮选择。

- `reports/o378_roof_v106_codegen/`：build.json、生成CUDA、完整SASS/resource、nvcc日志。
  源SHA与生成体可CPU复算；候选.so只保留于下载归档，不进入Git。
- `runs/o378_roof_v106_full24/`：results.jsonl、source_provenance.jsonl、源identity核验、
  environment.json、GPU快照、原始summary及本地全阶段 `analysis.json`。
- `runs/o378_roof_v106_{preflight,memcheck,synccheck}/`：编码/合成/guard验证结果；
  同名reports日志保留0-errors与测试范围，非全4096³所有kernel/racecheck验收。
- `tests/unit/test_roof_v106_evidence.py`：重放raw Event汇总、配对CI、来源、源SHA、
  旧编码对照、GEMM身份及既有原生INT4审计。

O7 A conversion配对吞吐+1.5753%，conversion total +1.0251%；输出逐位/MSE不变。
GEMM CUfunction完全相同，不能宣称GEMM加速；直接计时CV失败较多，Cold/steady未确认收益。
保留独立转换候选，不改变最佳GEMM、正式默认、原生扩展或5090；停止相邻查表变体。

下载完整归档：`tmp/o378_v106_complete.tar.gz`，SHA-256
`d865133a4ef23b558d314ae4e6398cdabc00b0568ba9dd3538fd25d1d43f8825`。
results.jsonl SHA-256：`3851b25d5f869aa563c14109f8e853d543f102258c4324706e3b45238fe1e65e`。
原生 `_sm80.so` SHA-256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

[方法、结果、MSE、波动与停止依据](../../o7_mx8_warp_lut_20261007.md)。

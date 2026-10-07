# v137 原始证据：O3 全 K warp 分路

[简明结论](../../o3_route_cohort_review_20261007.md)：两路原生INT4保持；24样本配对吞吐−23.80%，不采纳。

- `analysis.json`：从144条完整记录重新计算的配对结果、MSE、资源与验证范围。
- `index.json`：所有原始文本的路径/字节数/SHA256和二进制归档SHA。
- `reports/o378_roof_v137_o3_route_cohort_fixed_codegen/`：PTX、SASS、逐指令liveness、编译与资源报告。
- `reports/o378_roof_v137_o3_route_cohort_codegen/`：最初ADL编译错误，未隐去。
- `runs/o378_roof_v137_full24/`：完整200次Event序列、24样本×3轮×新旧两侧、环境/输入provenance、汇总。
- `runs/o378_roof_v137_validation/`：96项正确性、8项拒绝、2项坐标检查。
- `runs/o378_roof_v137_{synccheck,memcheck,racecheck}/`和相应log：各4种模式的混合路径安全检查。

本轮源码commit `8358d8ff3df7f3eba78eff4be4f92a9f16ac1f37`；CUDA候选编译commit `02a4030ae28d1743cbb8daf59d0913a71d97b0f5`。
未重新编译或更改正式扩展，SHA256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

含cubin/host库的完整归档在本地及A100项目内：`tmp/o378_v137_complete.tar.gz`。
SHA256：`a613004629b9d06abb9cb5f685fc9c71c0db221e50aa5b253dfd03d87a6fa259`。
Git只保存文本证据，不保存cubin/so。测试逐文件检查SHA，并重新计算全部配对统计；不删除CV失败或离群记录。

```bash
python scripts/freeze_o3_route_cohort_evidence.py \
  --archive tmp/o378_v137_complete.tar.gz \
  --sha256 a613004629b9d06abb9cb5f685fc9c71c0db221e50aa5b253dfd03d87a6fa259 \
  --output tmp/o378_v137_recomputed_evidence
python -m pytest tests/unit/test_o3_route_cohort.py tests/unit/test_roof_v137_evidence.py -q
```

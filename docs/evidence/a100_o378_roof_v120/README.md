# v120 原始编译证据：固定 CTA 异构调度

仅固定1/3 CTA四链、2/3八链，完整K循环之外做CTA一致分派；不改数学、source格式、
factor/guard、v73准备、tile/stage/warp或正式默认。先核对已有机制，非统一链数重测。

- 源码commit：`9d9b8a103caff66318da3e2f63c3023c92bb97f8`；已先GitHub推送，再A100 fetch/ff-only。
- 原控制完整SASS与v78一致，同候选entry每整数路径均双原生INT4/cg copy，不是probe匹配。
- 原383条循环，候选368/367；原静态八链，候选静态八/六链；均168regs。
- 候选每G128分别新增4/3条LDL.64；全entry32B stack、72B spill load/store不是动态流量。
- 预设结构及零热localgate失败，停止候选及相邻比例/seed/链数扫描，不迁移O3。
- **未运行候选GPU**：无Event/MSE、conversion/端到端、NCU/sanitizer验收；不能把未测写成0%提升。
- `index.json`冻结14份原文本/生成header/源码及环境记录，保留原SHA，不重写原输出。
- complete archive保存在本地和A100 `tmp/o378_v120_complete.tar.gz`，含CUBIN；binary不进入Git。
  SHA：`bfc4b0f98577a1594ea0cc7e95a5986de991955463ba5ce8c3fc9ec6725f1894`。
- 正式扩展SHA仍 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，未重编译。

```bash
python -m pytest tests/unit/test_o78_phase_mixed.py tests/unit/test_roof_v120_evidence.py -q
```

CPU回放核对SHA、Git源码、生成header、完整旧控制编码、两条整数循环及失败gate。
不是GPU正确性或性能测试。[候选分析与查重](../../o7_o8_cta_phase_mix_20261007.md)。

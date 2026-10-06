# v91：地址重物化候选，负收益，不采纳

仅优化 grouped O7/O8 的 A-copy 地址存活，不改量化、G128 factor、全 K 安全整数数学、
两路原生 INT4、八条 partial 链、CTA64×128×128、两 stage、guard/回退、v73 准备或正式默认。
采用公开 PTX `mad.wide.u32` + `cp.async.cg.shared.global`，不修改机器码。

编译先验：v78/v89 对照编码一致；候选 integer loop 为 379 条指令、0 LDL/STL，
168 regs、静态存活峰值 163；原 v89 为 381 条、2 LDL、峰值 166。
候选 ptxas 报告 0 stack / 0 spill load/store；对照 v78 自身也无热循环 spill。
这只提供进行 runtime 的依据，不是性能收益证明。

84 项小 M/N、全 K4096 的 stream/数值/非法输入/分组混合回退检查通过。
随后直接 24 个原始 FP16 trace 样本、三轮配对、warmup1000 / repeats200 / inner100；
源格式初始量化在计时外，GPU preparation 和 CUDA stream 全部相同。
每个样本分别核对 raw/prepared SHA、source provenance、payload、factor/guard、FP16 主参考和逐位输出。

|GEMM-only|v78 ms|v91 ms|配对吞吐变化|95% CI|CV≥3%：旧 / 新|
|---|---:|---:|---:|---|---|
|O7|0.474112|0.480256|−1.28%|[0.987207,0.988285]|0/72；1/72|
|O8|0.475648|0.485888|−1.26%|[0.987207,0.989339]|2/72；1/72|

288 条记录全部 finite FP32、逐位 v67/v78 一致、MSE 回归通过，MSE 差为 0。
O7/O5 MSE median/mean：0.005536172273439442 / 0.005053635851002639。
O8/O6 MSE median/mean：0.004411084910985704 / 0.00438137929907354。
O7 全部 CTA 整数路径；O8 的 `layer_24_o_proj` 只有原有 12 个 CTA 走安全回退。
没有删除高 CV 或较慢记录；不同样本/轮次的整体中位延迟之比不是配对 speedup。

不采纳、不扩展负收益四模式/NCU/sanitizer；无新 conversion/Cold/steady 数据，
有限正确性检查不替代 sanitizer，更不作为全部 4096³ 输入的安全证明。
最佳和默认不变，目标没有完成。

完整归档（含不提交 Git 的 CUBIN）：`tmp/o378_v91_complete.tar.gz`。
本地/A100 SHA256 均为：
`31b9e289b725d145336846e84c71d19e9e509b3874c755997f8aaa12a10eddfd`。
候选编译源码 commit `09c880f`；运行脚本 `7e70e05`。
正式扩展仍 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
实现本地提交并推送 GitHub 后，A100 通过校验 bundle fetch/ff-only merge；不修改 5090。

复现（仅在需要重测时；必须使用新的仓库内输出路径）：

```bash
python scripts/probe_o78_address_remat_codegen.py --output reports/new_v91_codegen
python scripts/benchmark_o78_address_remat.py --cubins reports/new_v91_codegen \
  --output runs/new_v91 --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python -m pytest tests/unit/test_o78_address_remat.py tests/unit/test_roof_v91_evidence.py -q
```

运行需要已验证的 v67/v73/v78/v89 基线 artifacts；本次无需重编正式扩展。
首次 evidence 单测的缺省零 LDL/STL 字段处理已修正：raw CSV/SASS 和分析值没变。

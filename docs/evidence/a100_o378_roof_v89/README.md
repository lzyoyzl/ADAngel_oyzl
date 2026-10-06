# v89 文本证据

结论见 [本轮简明报告](../../o3_o7_o8_grouped_cta_20261006.md)。

本目录完整保留首次/复测原始 JSONL、统计、GPU 监测、两份同 entry PTX/SASS/
资源/存活审计、O3 四模式与有限 sanitizer 记录；没有筛掉高 CV 或较慢记录。
Git 不提交 CUBIN/driver 二进制，它们在完整归档及 A100 原目录中保留。

完整归档：`tmp/o378_v89_complete2.tar.gz`，两端 SHA256 一致：
`a1e6e2d8027e99a76a461e01ba608f1c256d25d60e4bf42100897ceeccf41a6f`。

|候选|CUBIN SHA256|
|---|---|
|O3 grouped CTA|89918880f8cad38f8234993ff07064d5c8bb6f2c5a4996c3d443d6378366856bd|
|O7/O8 grouped CTA|d69dc4ad567d2fd96cb16d08636af7e4dd1f638cd4ed825b47e5dd4955ee90555|

正式扩展仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`；没有重新编译或切换正式默认。
候选/驱动源码 commit `18fd6d2`；复测验证器 `9e7f81a`。
源码本地实现并推送 GitHub，A100 使用校验 bundle fetch/ff-only merge。
可选 NCU 脚本 `6042df3` 本轮未采集新报告，不把旧 NCU 数据标成新候选的数据。

```bash
python scripts/probe_grouped_cta_codegen.py --kind o3 --output reports/new_o3_grouped
python scripts/probe_grouped_cta_codegen.py --kind o78 --output reports/new_o78_grouped
python scripts/benchmark_o3_grouped_cta.py --codegen reports/new_o3_grouped \
  --output runs/new_o3_grouped --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python scripts/benchmark_o78_grouped_cta.py --cubins reports/new_o78_grouped \
  --output runs/new_o78_grouped --samples 24 --rounds 3 --warmup 1000 --repeats 200 --inner 100
python -m pytest tests/unit/test_grouped_cta_probe.py \
  tests/unit/test_grouped_cta_retest.py tests/unit/test_roof_v89_evidence.py -q
```

输出须为不存在的仓库内路径。O3 四模式额外使用 `--rounds 1 --modes conversion_only compute_only cold steady_state`。
sanitize 命令使用候选 symbol filter，三种工具各完成 96+8+2 项小 M/N、完整 K4096 检查。
其范围不是对全部 4096³ 输入的安全证明；O7/O8 负收益候选未扩展四模式或 sanitizer。

最后一次只读身份核对首次漏掉 `import torch`，出现 `libc10.so` 加载错误；
补齐后第二次验证成功、正式扩展 SHA 未变。该核对不生成任何性能结果，
不涉及 CUDA kernel 或编译故障。两次核对记录均保留。

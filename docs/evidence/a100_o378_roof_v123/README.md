# v123：O8 packed FP6转换的完整证据

24样本×三轮×两路径×四模式，共576条记录、345600个原始Event值，不过滤。
同一个v78 GEMM函数；FP6→Q6/F2/RNE、scale、平方和、guard和输出逐位不变。

| 指标 | 原v73 ms | v123 ms | 配对吞吐变化 |
|---|---:|---:|---:|
| A conversion-only | 0.057508 | 0.040589 | +41.39% |
| Conversion-only total | 0.091290 | 0.074414 | +22.61% |
| Compute-only GEMM | 0.481280 | 0.481280 | 0.00% |
| Cold total | 0.580608 | 0.566272 | +2.63% |
| Steady total | 0.545792 | 0.529408 | +3.04% |

O8/O6 MSE median/mean：0.004411084910985704 / 0.004381379299073540。
新Cold/steady total各5/72、3/72条CV≥3%，原值全保留；不能声称严格全stage CV通过。
保留独立转换候选，不改默认、正式扩展、O3/O7/5090。没有GEMM优化收益。

`index.json`列出31份原始文本的SHA/字节数与编译/运行commit；二进制只在本地与A100归档：

```text
tmp/o378_v123_complete.tar.gz
SHA256 6d6f83db3d3c5ba52bc5c04d6f9c368c76d92ccedac147bd06d4d33bcdfddbde
```

第一次编译的“不允许寄存器增加”gate失败，候选未运行。随后资源审查确认23→29regs并未改变实际8CTA/SM，才进入GPU测试；两次完整编译结果均保留，不能删除最初门槛失败记录。
新旧CUDA entry696→408静态指令，四条原生DP4A，shared256B/一barrier/零local；所有旧入口编码不变。

131072 word的GPU/CPU编码及平方和验证、32项合成四模式和3项guard边界通过。
memcheck/synccheck仅检查两个新增转换entry、有限小M/N及K4096，均0errors；不声称完整GEMM或racecheck重新验收。

```bash
python -m pytest tests/unit/test_nv6_swar_codegen.py tests/unit/test_roof_v123_evidence.py -q
```

[实现、瓶颈、完整结果和平台迁移](../../o8_packed_fp6_conversion_20261007.md)。

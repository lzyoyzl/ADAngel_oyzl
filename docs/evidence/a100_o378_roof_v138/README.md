# v138：HiF4 packed转换，24样本四模式配对

只改O8权重转换；双方v123 FP6激活转换和v78两路原生INT4 GEMM相同。
G128有效scale、精确平方和/guard和量化语义不变；无INT8 Tensor Core替代。

| 指标 | 旧 ms | 新 ms | 配对吞吐变化 |
|---|---:|---:|---:|
| W conversion-only | 0.033802 | 0.018540 | +83.17% |
| Conversion-only total | 0.074440 | 0.059110 | +26.25% |
| Compute-only GEMM | 0.481280 | 0.481280 | 0.00% |
| Cold total | 0.566272 | 0.549888 | +2.98% |
| Steady total | 0.529408 | 0.529408 | 0.00% |

24样本×3轮，warmup1000/repeats200/inner100；576条完整记录、345600原始Event值。
输出逐位一致，vs O6 MSE median/mean仍0.004411084910985704 / 0.004381379299073540。
新conversion-only W/total均0/72 CV≥3%，Cold total2/72；但Cold内W分项54/72超标，不能宣称全stage稳定。
所有离群值保留。`layer_24_o_proj`既有12 CTA回退，其他23样本全整数；两侧guard完全相同。

原进程SSH断开后终止。`runs/o378_v138_full24`保留原483条；`resumed4`补测后4样本；
`completed24`由原480条完整样本记录+新96条组成。原末尾3条未完成样本记录仍在归档，
不按速度/CV删点；`recovery.json`记录源SHA及采集commit。

HiF4入口静态指令544→312、寄存器31→22、资源8CTA/SM不变、shared256B、零local。
四条标量DP4A用于精确平方和，不是INT8 Tensor Core；所有21个旧入口编码不变。
最初匿名probe符号哈希审计失败被保留；限定唯一符号归一化后比较完整机器字，通过。
1,048,576 GPU decoder组合、32合成/3边界通过；定向转换memcheck/synccheck小M/N、K4096均0errors。
不声称新做完整4096³ GEMM内存安全审计。

文本及逐项SHA在`index.json`，二进制只保留于本地/A100归档：

```text
tmp/o378_v138_complete.tar.gz
SHA256 b16339ebd987e4bd6365564ffa7b46011f5e33f84565b27566f198ed10229b18
```

```bash
python -m pytest tests/unit/test_hif4_resume.py tests/unit/test_hif4_swar_codegen.py \
  tests/unit/test_roof_v138_evidence.py -q
```

[实现、数据格式、结果与限制](../../o8_hif4_packed_conversion_20261008.md)。
正式扩展/默认、O3/O7 GEMM和5090不变。

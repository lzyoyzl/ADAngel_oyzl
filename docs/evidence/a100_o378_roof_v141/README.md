# v141：受限系数窗口，完整双次24样本证据

只改独立O7/O8全K候选的寄存器生命周期和系数消费顺序，未改默认、转换或5090。
编译前门槛通过不等于性能验收通过：383→372条静态指令、168regs、零热local，
同entry保留U4×S4/S4×S4；64项依赖变为独立系数乘积加一次更新。

两次同配置24×2 variants×3 rounds×2 policies，各288条记录，每条200次GEMM计时。
首轮O7/O8配对吞吐−1.883%/−1.982%；重测−1.953%/−1.938%。
全部记录CV≥3%，约53%–55%的中位CV，存在其他用户训练进程；不声称严格稳定性通过。
两次均逐位输出相同，MSE不变。未过滤、未锁频、未等待空闲或操作其他进程。
无采纳依据，当前最佳仍O3 v89/O7-O8 v78；无新conversion/端到端/NCU成绩。

## 文件

- `reports/o378_roof_v141_recycled_coefficient_codegen/`：源哈希、PTX/SASS、活跃寄存器、资源、编译gate和定向sanitizer日志。
- `runs/o378_roof_v141_paired/`及`_r2/`：每个原始计时、输入provenance、GPU快照、MSE、guard及运行环境。
- `runs/o378_roof_v141_validation/`、`_memcheck/`、`_synccheck/`：64合成/模式＋12边界检查；
  定向候选entry、小M/N、K4096。不是整个4096³的sanitizer验收。
- `reports/o378_roof_v141_gpu_concurrency.txt`：测试间隙的GPU状态与其他compute PID。
- `analysis.json`：逐Event重算和逐值SASS交错审计；旧16次更新/7条新MMA→候选1/1，是静态计数，不是周期数。
- `index.json`：全部文本SHA、二进制SHA、编译/运行提交。

编译提交`5a1ac427e4bf7ce9434be2afea8b68120435247c`，运行提交`47f848e`（完整值见index）。
正式扩展运行前后SHA为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
完整二进制归档`tmp/o378_v141_complete.tar.gz`（未提交Git），服务器/本地SHA一致：
`cf50963c992e12f897efb7a4aa39227a02bfb56c4a6b953028718ab0118028f6`。
首次memcheck调用把NCU的`regex:`语法误用于sanitizer，CLI退出、未运行候选；改成`kne=`后0 errors。

## 无GPU复算

```bash
python -m pytest tests/unit/test_recycled_coefficient_codegen.py \
  tests/unit/test_roof_v141_evidence.py -q
python scripts/analyze_o78_recycled_coefficient.py \
  --input docs/evidence/a100_o378_roof_v141/runs/o378_roof_v141_paired_r2 \
  --output tmp/v141_recomputed_fresh.json
```

[结果与实现说明](../../o7_o8_recycled_coefficient_20261008.md)。目标尚未达到，不把负结果当作性能上限证明。

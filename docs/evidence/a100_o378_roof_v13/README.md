# v13：三阶段流水线的寄存器预算隔离

候选18/19与16/17分别保持相同的K128三阶段计算主体，仅将`__launch_bounds__(256,3)`改为`(256,2)`。**结论：明显缓解O3退化，但未超过双缓冲候选6，不采用为默认。**

## 版本与正确性

- GPU：A100-PCIE-40GB，CUDA12.8；构建/测试源码`0a3f75c`。
- 扩展SHA-256：`ab0d3c9c1c0bc9cadaf0042c0d3bc2ce8f0369a19b52a5f77c34e77d9860acb2`。
- 12个原生产实例、45个旧候选的编码SASS与v12逐字一致；51个候选实例的原生U4×S4/S4×S4和cp.async审计通过。允许spill策略未隐去原始检查：18/19的O7/O8无LDL/STL，O3仍各一处。
- 522项合成逐位检查通过，包含随机、零值、饱和值、不同scale及非默认stream；K128/384/640覆盖三阶段排空。
- memcheck、synccheck各252项，均0错误；K768三后端racecheck为0 hazards/errors/warnings。这不是任意输入的形式化证明。
- `layer_00_q_proj`四模式48条记录全部与旧实现逐位相同。未因候选较慢而扩大为24样本正式验收。

## 十轮同二进制合成初筛

4096³，warmup50、repeats200；五个实现循环轮换，每个位置各出现两次。下表为各轮median的median，单位ms；不是真实trace最终结果。

| 后端 | 原正式 | 候选6 | 候选17 | 候选18 | 候选19 |
|---|---:|---:|---:|---:|---:|
| O3 | 0.558080 | 0.536576 | 0.971264 | 0.603136 | 0.601088 |
| O7 | 0.601600 | 0.558592 | 0.596480 | 0.593920 | 0.593920 |
| O8 | 0.604160 | 0.562176 | 0.600064 | 0.594944 | 0.595968 |

CV≥3%的轮数/10，按上表列顺序：O3为`4/5/1/2/2`，O7为`3/2/5/3/3`，O8为`1/5/2/3/5`。不删除离群轮，不用GPU快照武断归因；不宣称全阶段稳定性验收通过。

## 同二进制NCU诊断

合成输入、9个section、clock/cache control=none，每种候选单kernel采集；未锁频。下表不是CUDA Event性能表，服务需求使用1410MHz参考时钟，不是可达时间预测。

| 指标 | O3/6 | O3/19 | O7/6 | O7/19 |
|---|---:|---:|---:|---:|
| NCU duration ms | 0.467584 | 0.524768 | 0.489120 | 0.516064 |
| 动态指令数 | 117,268,480 | 125,894,656 | 127,238,144 | 128,892,928 |
| Eligible warps/scheduler | 0.787 | 0.757 | 0.891 | 0.801 |
| Issue active % | 45.62 | 41.51 | 47.22 | 44.06 |
| 理论local sectors | 6,291,456 | 131,072 | 10,485,760 | 0 |
| 额外shared wavefronts | 0 | 8,388,608 | 0 | 8,388,608 |
| L1服务需求 ms | 0.274863 | 0.278892 | 0.288158 | 0.276450 |

三个案例的IMMA/I2F/FFMA工作量均保持16,777,216条warp指令。O7/19不存在local指令，NCU不导出local-sector列；解析器只有在同一source/SASS确无LDL/STL时才将该缺省计为0，并显式标注，不将任意缺失指标当作0。复算脚本版本`65236c6`；采集与二进制仍为上述`0a3f75c`。

三阶段19的额外shared工作全部定位于LDGSTS，LDSM额外wavefront为0；其LDSM本身的工作量仍为25,165,824。K128使BAR/DEPBAR由262,144增至524,288。O3的local访问已减少约98%，O7则完全消除，但发射率下降、指令数量未减少，实际仍较慢。因此**spill减少不等于整体加速；加深pipeline也没有降低主要fragment供数工作**。

下一步改为减少跨M warp重复B读取，保持K256双缓冲，并同时观察寄存器增长与就绪warp减少的代价。候选20/21尚待GPU验收，不能由本报告推定其收益。

## 单真实样本MSE与计时

所有实现、四种模式的MSE相同，候选相对原实现MSE均0：

| 后端 | 相对O0 MSE | 组内参考 | 相对组内参考MSE |
|---|---:|---|---:|
| O3 | 0.0005745973478203796 | O0 | 0.0005745973478203796 |
| O7 | 0.005262686510301441 | O5 | 0.00017169781257259873 |
| O8 | 0.0034977923595691848 | O6 | 0.00021370230554795398 |

转换代码未变。转换阶段inner100摊销，cold/steady total按单次Event直接计时；每阶段原始样本、CV及MSE均在`runs/o378_roof_v13_four_smoke/results.jsonl`。以上单样本值不能冒充24样本median。

## 文件与复算

`reports/o378_roof_v13/`保存审计、控制组SASS对比、NCU导出、对照JSON及安全检查日志；`runs/`保存原始Event样本、环境、校验与trace provenance。

完整PTX/SASS、`.ncu-rep`及v13扩展备份`audited_extension.so`留在A100项目的`reports/o378_roof_v13/`，不入Git。传输归档`tmp/o378_roof_v13_evidence.tgz`的SHA-256为`230bdfbb8feb40b0b13456ea3f3381a2549c91fd275d8e109a532ca1ef0f58b7`，本地与远端一致。

```bash
python scripts/analyze_roof_scale_ncu.py \
  --directory docs/evidence/a100_o378_roof_v13/reports/o378_roof_v13 \
  --variant o7 --tunes 6 17 19
python -m unittest discover -s tests/unit -p 'test_roof*.py' -v
```

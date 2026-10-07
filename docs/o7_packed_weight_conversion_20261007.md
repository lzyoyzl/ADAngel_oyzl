# v118：packed NVFP4 权重转换候选

本轮已完成编译、正确性、24样本四模式配对和有限安全检查。
这是新的权重 packed 布尔解码结构，不重复已测试的 GEMM 优化。
确认O7转换与Cold的小幅收益；GEMM没有提升，正式默认、原扩展、O3/O8、trace、源量化及5090保持不变。

## 实质区别与范围

| 已完成方向 | 本轮区别 |
|---|---|
| v34 标量整数 RNE 解码 | 旧代码逐元素取4-bit code、整数查表、符号处理、packing；本轮一次处理一个32-bit word里的8个元素 |
| v53 16元素向量读写 | 沿用原 uint2 读写，不把既有向量加载当成新优化 |
| v69/v73 平方和与 row metadata 融合 | 完整保留原融合结构、G128、factor/anchor/norm/guard；只改变 packed 解码与精确平方和的计算 |
| v106 MXFP8 warp 分布式查表 | 不重做激活查表；本轮是权重的寄存器位并行布尔计算，无新增表或 payload shuffle |
| tile/stage/warp、累加链、供数重排、scale提前、magic及固定high移位 | 全部不改、不重测 |

这是 **conversion/Cold 候选**，不是 GEMM 优化。两种准备路径必须使用同一个
v78 GEMM CUfunction，不能把 Compute-only 的频率/调度波动归因于这次修改。

## 精确算法

E2M1 的8种正幅值按原 F=0/RNE 转 Q4，仍为
`{0,0.5,1,1.5,2,3,4,6} → {0,0,1,2,2,3,4,6}`。
抽取一个 word 中8个 nibble 的三个幅值 code plane 与符号 plane，使用公开
`lop3.b32` 的固定真值表同时计算8个 Q4 幅值，再在各 nibble 内转二进制补码。
负零和负0.5均得到整数0。每个 nibble 的加法中间值不超过8，不产生跨 nibble 进位。
输入位与真值表的顺序按 CUDA 12.8 官方PTX定义核对，不使用未公开机器码：
[LOP3 指令定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3)。

范围 guard 需要精确平方和，不能省略。平方值为
`{0,0,1,4,4,9,16,36}`；对其五个非零 bit plane 做 POPC，并按位权求和，
得到与8次 `q*q` 相同的整数值。其后 subwarp reduction 与所有 metadata 运算
逐字沿用v73。没有近似平方和、截断、改变 scale 或放宽整数/FP32安全 guard。

## 预先规定的投入门槛

先穷举所有16编码、全部四-nibble组合（重复/互补两种组合），再检查随机完整 word。
只有以下编译条件全部满足才投入 GPU 验证和直接24样本四模式配对测试：

- 同一旧库入口完整编码 SASS 与 v73一致；
- 新权重转换静态指令至少减少5%；寄存器不超过原31；
- shared仍256B、一个 CTA barrier、stack/local/spill均为0；
- 精确平方和包含每两个输入word共10次POPC，不换成近似范围估计。

静态减少比例不是延迟收益承诺。原v106同进程W约0.0222ms、Cold约0.5530ms，
权重转换即使完全消失也只是约4%的粗略机会量级；两种计时口径不同，不能严格相减。
本轮必须如实报告转换与端到端配对结果，若负向或收益不足则停止该路线，不扫相邻布尔/查表变体。
GEMM逼近有效吞吐上界的主要目标不因这个辅助候选而改变。

```bash
python -m pytest tests/unit/test_nv4_swar_codegen.py -q
python scripts/probe_nv4_swar_codegen.py --output reports/o378_roof_v118_codegen
python scripts/benchmark_o7_nv4_swar.py --validate-only --output runs/o378_v118_preflight
python scripts/benchmark_o7_nv4_swar.py --samples 24 --rounds 3 \
  --warmup 1000 --repeats 200 --inner 100 --output runs/o378_v118_full24
python scripts/analyze_nv4_swar.py --input runs/o378_v118_full24 \
  --output runs/o378_v118_full24/analysis.json
```

源码先在本地提交并成功推送，再在A100项目内fetch/ff-only、编译。计时仍使用原双轨
CUDA Event方法。A100编译 gate 已通过：392→320指令（−18.367%）、31→30寄存器，
256B shared、一个barrier、10次POPC、stack/local=0，旧入口完整编码SASS一致。
这是静态成本减少，不是18.367%的延迟提升。

## 完整24样本结果

编译commit `0891b564`，运行commit `0f006fc5`；均先本地推送，再在A100项目内
fetch/ff-only。24真实FP16 trace直接产生相同O7源格式，identity逐字段与v99一致。
三轮交错A/B，warmup1000、repeats200、conversion inner100，单stream、预分配。
共576记录、345600原始Event值，不删除离群值，不挑最快轮次。未锁频、未等待其他GPU任务空闲。

ms为各样本三轮median再跨24样本取median；配对吞吐由同样本同轮旧/新延迟比
独立汇总，不能直接除下表两列median代替。bootstrap按24样本重采样10000次，seed20261007。

| 指标 | 原v73 ms | v118 ms | 配对吞吐变化 | speedup 95% CI | CV≥3%旧/新（各72条） |
|---|---:|---:|---:|---|---:|
| Conversion-only W | 0.022205 | 0.018217 | **+21.82%** | 1.216418–1.221938 | 0/0 |
| Conversion-only total | 0.067297 | 0.063165 | **+6.42%** | 1.062939–1.065260 | 0/0 |
| Compute-only GEMM（同一kernel） | 0.481280 | 0.481280 | 0.00% | 1.000000–1.000000 | 2/0 |
| Cold total | 0.558080 | 0.554496 | **+0.65%** | 1.005535–1.007401 | 0/0 |
| Steady-state total | 0.534528 | 0.534528 | 0.00% | 1.000000–1.000000 | 0/4 |

转换总配对延迟下降约6.03%，Cold约0.64%；表中的+6.42%/+0.65%是吞吐变化。
转换仍单独批量摊销，Cold/steady total仍是单次直接Event，不能相减拼装性能。
完整13项stage统计在analysis.json，A转换无确认收益；本次没有修改A或GEMM。

### 波动与验收边界

Cold的隔离批量W阶段旧/新CV失败 **6/57**，虽median仍约0.022323/0.018253ms，
新CV min/median/max为2.27%/3.20%/3.85%。最多仅略高于3%的预定门槛，
但仍必须保留为失败，不更改阈值，也不声称所有正式阶段稳定验收通过。
同kernel的独立conversion-only W和直接Cold total各72条均CV<3%；Cold的GEMM也0/0失败。
因此可报告完整配对分布的正向趋势和直接Cold收益，但不能将单独Cold W阶段说成严格稳定。
采样SM clock1245–1410MHz；未做本轮NCU或逐条因果测量，不能把波动全部归因某一具体原因。

## 正确性与审计

| 检查 | 结果 |
|---|---|
| CPU packed计算 | 全16编码、四-nibble穷举重复/互补、16384随机完整word与独立RNE参考一致 |
| GPU word检查 | 131072项scalar/SWAR/CPU packed值及精确平方和一致，非默认stream |
| 合成/边界 | 32项四模式与9项原非法编码/范围/零值guard通过，candidate准备同样覆盖guard |
| 24样本数值 | 全576条finite FP32、与v67逐位相同、输出间MSE和max abs差均为0；packing、scale、平方和scratch每次运行后检查 |
| O7相对O5 MSE median/mean | **0.005536172273439442 / 0.005053635851002639**，没有变化 |
| GEMM | 两policy同一v78 CUfunction/CUBIN：168regs、零local、3 CTA/SM、64×128×128、2stage；原生U4/S4及S4/S4审计沿用已校验原binary |
| 转换编译 | 新entry320指令/30regs；旧v73入口完整编码SASS不变 |
| memcheck/synccheck | 两个新的转换entry过滤、有限small-M/N/K4096和word检查均0 errors；非完整GEMM/4096³/racecheck验收 |
| 正式扩展 | SHA保持94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462，未重编译 |

## 决定与insight

保留为独立O7权重转换候选，不自动修改production默认。静态工作减少能够带来
实际W转换收益，但W原本仅约22微秒，GEMM约481微秒；减少约4微秒后，
直接Cold只提升约0.65%。steady缓存W，所以没有收益。这是开销占比的限制，不是MSE代价。

本轮不把转换优化冒充GEMM吞吐突破；O3 v89和O7/O8 v78的GEMM最佳不变，
接近有效吞吐上界的主目标尚未完成。停止该路线的相邻布尔/查表扫描，不继续追逐微小端到端差别。
[24份原始文本、完整归档SHA与CPU复算](evidence/a100_o378_roof_v118/README.md)。

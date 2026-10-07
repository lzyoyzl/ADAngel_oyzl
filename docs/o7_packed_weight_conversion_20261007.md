# v118：packed NVFP4 权重转换候选

本轮先做 CPU 精确性与编译成本 gate，不重复已测试的 GEMM 优化。
正式默认、原扩展、O3/O8、trace、源量化及5090保持不变。

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
这是静态成本减少，不是18.367%的延迟提升；目前尚无 GPU 性能、MSE、安全性或端到端新结论。

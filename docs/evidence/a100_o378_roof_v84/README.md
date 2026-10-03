# v84：全K整数路径的三阶段前瞻，独立候选

状态：本地实现与CPU源契约/槽位模型已检查，A100编译、运行和性能结果待验证。
不修改正式默认、O3或5090，不将这份设计说明计作性能收益。

## 测试依据与唯一变化

v80显示O7/O8仍有供数/发射等待；v83扩大tile虽减少加载，却增加寄存器并降低驻留CTA，未获益。
这里保留v78的64×128×128、四warp、八条MMA链和全K整数guard，仅将整数路径从两槽改为三槽，
以相同每组加载工作量更早发出下一组数据。不重启旧FP32路径的stage/tile扫描。
原先每轮只有下一组前瞻，本候选先提交group0和1；处理group g时允许g+1仍在途，预取g+2。

`cp.async.wait_group 1`等待除最新一组以外的拷贝完成；末组用`wait_group 0`排空。
随后保留CTA barrier，保证跨线程可见性，并确认group g−1的读者结束后才能复用其槽位。
不能简单删除wait或barrier。依据：[NVIDIA PTX等待语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)。

shared从34304增至51456B，launch bound仍128线程/3CTA。实际驻留与spill以编译和Driver查询为准。
不新增payload/scale量化、系数操作或输出；unsafe CTA保留原两阶段逐组FP32回退。
转换与范围证明仍为v73，不能将新增guard准备排除在端到端时间外。

## 验收与命令

先核对旧控制的机器码完全一致、同entry原生U4/S4与S4/S4、cp.async及wait1/0、
MMA/LDSM工作量和资源；再运行非默认stream的数值/边界与有限sanitizer。
四样本×三轮配对初筛，全部Event/CV失败保留；没有收益则停止，有明确收益再扩大24样本与四模式。

```bash
python scripts/probe_o78_three_stage_codegen.py --output reports/o378_roof_v84_codegen
python scripts/benchmark_o78_three_stage.py --cubins reports/o378_roof_v84_codegen \
  --output runs/o378_roof_v84_screen --samples 4 --rounds 3 --warmup 50 --repeats 200
```

需要现有v67/v73/v78构建和原FP16 trace；输出目录必须全新。
CPU槽位模型覆盖更多尾长，但实际GPU候选仍只支持K4096；不能据此声称其他K的运行验收。

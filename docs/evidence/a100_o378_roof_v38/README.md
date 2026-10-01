# v38：显式L2预取初筛——未确认性能收益

**结论：不采用。O3最佳仍为54，O7/O8仍为59；正式扩展、转换路径和RTX5090代码不变。**
本轮在保留`cp.async.cg`的条件下，测试L2预取128B/256B，未发现可确认的加速。

## 方法与审计

复用54/59的原device body，只修改异步拷贝的L2预取提示，编译为独立cubin。
依照[CUDA12.8 PTX文档](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async)，
这是缓存预取提示，不改变16B的实际shared-memory拷贝量或同步语义。

- 无hint控制的编码SASS，与当前54/59完全相同；O3有944条静态指令，O7/O8有880条。
- 128B/256B确实编译为`LDGSTS.E.BYPASS.LTC128B.128` / `LTC256B.128`。
  除此标记外，指令文本、顺序、寄存器和操作数相同；不是声称三者机器码完全相同。
- 六个entry均保留原生U4×S4、S4×S4 IMMA，没有INT8替代。
- 仍为168寄存器/线程；O3 stack16B、O7/O8 stack8B。没有消除原有spill。
- CTA64×128×128、4warp、G128独立scale和FP32输出不变；O3三阶段，O7/O8两阶段。

三种cubin都由同一个原生CUDA Driver/Event循环计时；事件在计时前分配，单次launch直接计时。
这避免Python逐次发射对短kernel计时的干扰，但仍是**内部compute-only初筛**，
不是新增正式后端或四模式验收，不将其绝对延迟与旧扩展run跨运行相除。

## 24个真实样本结果

24样本×3轮，warmup50/repeats200，共648条记录。同样本同轮循环换序；不删除CV失败或离群记录。

|后端|无hint median ms|128B median ms|128B配对吞吐变化|256B median ms|256B配对吞吐变化|
|---|---:|---:|---:|---:|---:|
|O3|0.476160|0.477440|−0.21%|0.477184|−0.11%|
|O7|0.501760|0.500736|+0.05%|0.501760|0.00%|
|O8|0.506880|0.506880|−0.10%|0.506880|0.00%|

吞吐变化为同样本同轮速度比，先在样本内汇总轮次，再跨样本汇总；不是直接用表中两列median相除。

|后端|128B speedup描述性95%区间|256B speedup描述性95%区间|CV≥3%条数：无hint / 128B / 256B|
|---|---|---|---|
|O3|[0.997768,1.000000]|[0.995717,1.001071]|23 / 25 / 22|
|O7|[1.000000,1.004053]|[0.995910,1.002049]|23 / 23 / 24|
|O8|[0.992095,1.001006]|[0.994955,1.002020]|24 / 24 / 24|

每项CV条数的分母为72。区间均包含1，**没有确认收益**，不能据此宣称O7加速0.05%。
共享、未锁频GPU；样本同属一份trace，bootstrap只作描述性比较，不是独立任务总体置信结论。

## 正确性与MSE

所有648次输出均为finite FP32，并与对应最佳54/59逐位一致。
沿用原始FP16 trace直接量化的流程，检查raw/prepared的SHA及公共准备结果逐位重放，未二次量化替代原始数据。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|L2策略引入的MSE变化|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|0，输出逐位相同|
|O7 / O5|0.005536172426666|0.005053635833762|0，输出逐位相同|
|O8 / O6|0.004411084948645|0.004381379215302|0，输出逐位相同|

参考输出不同，不宜用这些MSE对格式作普遍精度排名。
单样本初测通过；memcheck和synccheck分别在一个真实4096³样本、三后端、三提示上通过，均0 errors。
这只是有限安全检查，不冒充穷举所有shape。相关CPU测试245项通过，原始日志保留。

## 瓶颈启示与下一步

v37的L1路径明显退化；本轮保留cg后改变L2预取，也没有可确认收益。
因此当前优先级不再是继续枚举缓存提示，而是减少真实指令工作、改善MMA/后处理之间的依赖与发射调度。
本轮没有减少MMA、I2F或scale工作，**没有确认缩小与约0.220347ms固定工作容量下界的差距**。
该下界仍不是可保证达到的延迟；不能把未发射采样占比当作可直接减去的耗时。
本轮未新采集NCU，不将前一轮的访存计数冒称为128B/256B的观测结果。

## 版本与复现

- Probe源码：`6c25eaa4882b5d7f35ca4136ccc0885921b33258`。
- Event初筛：`5b0863e5cf43036d86167da25616315e93e8a01f`；加强审计/汇总：`d9e1b34c1d1bcc40aa4a77c23e0309930ba111e1`。
- 正式扩展SHA-256始终为`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`，没有重新构建或覆盖。
- `reports/o378_roof_v38/`包括编译命令、版本、资源、四份精简SASS及可重算的审计。
- `runs/o378_roof_v38_trace24/`保存Event原始采样、MSE、输入provenance与GPU快照。
  smoke/memcheck/synccheck的数据不混入性能汇总。
- 三个cubin、完整PTX和driver `.so`留在A100项目的对应路径；Git归档是47个文本证据。
  传输归档SHA-256：`2c58100c3cdce49116f6a68e026f80d3b67d18717dff0b631f3f676d2dfac7d9`。

```bash
python scripts/probe_roof_l2_codegen.py --output reports/l2_recheck
python scripts/benchmark_roof_l2_probe.py --cubins reports/l2_recheck \
  --output runs/l2_recheck --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/audit_roof_l2_probe.py --directory reports/l2_recheck \
  --best-sass reports/o378_roof_v37/audit/extension.sass
```

需要当前已验证的A100扩展与trace；输出目录必须全新。本轮候选没有加入正式运行入口。

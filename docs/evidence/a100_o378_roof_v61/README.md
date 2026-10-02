# v61：O3 在 GPU 内选择安全路径，保留有限收益

**24 样本确认：相对 O3/54，缓存后的 GEMM 配对吞吐 +3.18%；计入 GPU guard/factor 准备后 +2.42%。**
相对上一版 v59 的缓存 GEMM，本版仅 +0.44%，不是新的大幅加速。
本轮只测试一个候选；O7/O8、完整端到端最佳组合及正式默认均不改。

## 改了什么

沿用 v60 的 GPU 范围检查与 factor 生成，以及 v59 的全 K 整数累加数学。
将每次 CPU 回传、同步后选择 kernel，改成同一 GEMM 内按 N128 tile 选择：

```text
GPU 准备：W scale → 精确对齐 factor / anchor / 每 N128 一个状态
GEMM：整个 CTA 读取同一状态
    安全 → 全 K INT32 流式累加，最后转 FP32
    范围检查不通过 → 旧 O3/54 逐 G128 FP32 路径
    非法 scale → 该 CTA 不写输出；批次完成后主机检查状态并抛错
```

两次 kernel 在同一 stream 排序。非法输入必须在输出交给 Python 调用方之前拒绝；
不是忽略错误或把未写入的输出当作正确结果。不同 N128 tile 可分别走快、慢路径，
但同一 CTA 内分支一致，不破坏内部同步。仅支持当前探针的 K4096、M64/N128 对齐形状。

安全界仍为 `131072 × sum(2^(code−anchor)) <= INT32_MAX`，保证所有乘积和前缀安全。
只接受正常 UE8M0 code 1..254；255 拒绝，0 为当前独立探针未支持并明确拒绝的情况。
源量化、独立 G128 scale、两路原生 INT4 和 FP32 输出保持不变。

## 配对性能

A100，4096³；24 个真实样本 × 3 轮，50 预热、200 次单次 CUDA Event，循环换序。
不锁频、不删除离群。表中时间为每样本跨轮 median，再取样本 median；
speedup 先按样本/轮次配对，因此不能直接用表内两个总体 median 相除。

|本轮独立计时口径|median ms|相对 O3/54 配对吞吐变化|speedup 95% CI|CV≥3% / 72|
|---|---:|---:|---|---:|
|控制 O3/54，仅 GEMM|0.476672|—|—|17|
|上一版 v59，准备/主机决策在计时外，仅 GEMM|0.465920|+2.65%|[1.021930, 1.029680]|17|
|v61，准备在计时外，GEMM 内 GPU 选择|0.463872|**+3.18%**|[1.026726, 1.035165]|16|
|v61，GPU 准备 + GPU 选择/GEMM|0.465664|**+2.42%**|[1.022026, 1.026432]|18|

v61 缓存 GEMM 相对同轮 v59 缓存 GEMM：**+0.44%**，CI `[1.001114, 1.006593]`。
v61 准备+GEMM 相对同轮 v59 缓存 GEMM：−0.11%，CI `[0.995604, 1.002188]`，未确认差异。
后两项从相同原始配对记录计算，不把不同轮次的最好值拼在一起。

前置四样本初筛的相对 O3/54 收益分别为 +3.25%（缓存）、+2.46%（含准备），与确认方向一致。
初筛相对 v59 的缓存收益约 +0.65%，区间跨 1；不以初筛单独认定胜出。

**这四行不是正式 conversion/compute/Cold/steady 四模式。** A/W payload 已完成原有转换与布局处理。
末行包含新增 GPU 准备，却不包含原有 A/W 转换；状态回传、错误检查在计时批次完成后进行。
所以它不是完整 Cold、steady 或主机 API wall time。没有新增完整 conversion/Cold/steady 数据。
全部 CV 失败保留，不能宣称严格 CV<3% 验收；bootstrap 区间为关联 trace 样本上的描述性比较。

## 正确性与 MSE

|24 样本输出指标|结果|
|---|---:|
|MSE / O0 median|0.006653010287409885|
|MSE / O0 mean|0.007578847013302749|
|新旧输出 MSE|0|

288 条确认记录的输出与 O3/54 逐位一致，metadata 逐项一致；初筛 48 条也一致。
每次预检包括 56 项合成检查、12 项非法 scale 拒绝及 6 项原地修改 scale 后恢复检查。
覆盖随机、零、极值、零行 scale、安全宽指数、unsafe 回退、同一矩阵内安全/不安全 N128 tile，
使用非默认 stream；与 FP64 语义参考及精确整数 guard oracle 对照。
memcheck/synccheck/racecheck 分别重跑这些检查，均 0 errors，racecheck 另 0 warnings。
sanitizer 范围是 M/N=64×128、128×256，K4096；不是完整 4096³ sanitizer 覆盖。

## 指令、资源与结论边界

同一正式候选 entry 的 PTX/SASS 包含 `U4×S4` 与 `S4×S4` 原生 INT4 MMA 及 `cp.async.cg`，
没有 INT8 MMA 替代。O7/O8 sentinel 编码 SASS 不变，原生扩展二进制不变。

|资源|O3/54|v59|v61 GPU 选择|
|---|---:|---:|---:|
|寄存器 / 线程|168|168|168|
|驻留 CTA / SM（实际配置）|3|3|3|
|dynamic shared / CTA，B|50688|50688|50688|
|Driver local size / 线程，B|16|40|32|

v61 ptxas 报告 52/52 B spill store/load，v59 为 36/36 B；v61 同时包含两条分支，
不能把静态 spill 总数或 2064 条完整函数静态指令当作一次快路径的动态执行量。
本轮未追加 NCU，不能将额外 +0.44% 精确归因于某项 stall 改善。

本轮主要确认：**GPU 准备本身的成本可以保留在组合计时中，同时避免每次主机决策的同步间隙。**
它没有减少 MMA 必要工作，距离约 0.220347ms 的理想容量下界仍明显，不能宣称解决核心吞吐瓶颈。
收益有限，不继续扫描分支写法、tile 或 cache。保留独立候选；完整最佳仍为 O3/54+转换2、
O7/O8/59+转换5，正式默认和 RTX5090 不变。后续若集成，需对权重版本缓存与完整端到端另行验收。

## 复现与证据

实现提交：`e3d4182`（编译审计）、`e14e43d`（测试），均先本地/push，再 A100 fetch/ff-only merge。
复用已审计 v59 GEMM cubin 与 v60 preparation cubin；只新增独立 cubin/Driver，不重编译正式扩展。

```bash
PYTHONPATH=python python scripts/probe_roof_device_factor_codegen.py --output reports/v61_rebuild
PYTHONPATH=python python scripts/benchmark_roof_device_factor_probe.py \
  --device-cubin-dir reports/v61_rebuild --output runs/v61_recheck \
  --samples 24 --rounds 3 --warmup 50 --repeats 200
```

输出目录必须不存在。完整原始 Event、环境、输入 hash、资源、SASS/PTX、编译与安全日志随本目录保留。
本地证据测试重新计算汇总/CV/配对区间并检查样本完整性、ISA、资源和安全日志。
[24 样本汇总](runs/o378_roof_v61_trace24/summary.json)；[初筛](runs/o378_roof_v61_screen/summary.json)。

候选 cubin SHA256：`d390ad87e2bfb4f3326785482cfebd2c67f36884854c904e41356e30551b9e8c`。
原扩展 SHA256：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
47 个原始证据文件的传输归档 SHA256：`1b7fe7e7d190a2c8cf0d49edfb0aa3b8d361edfb78650c4b1ca132b770371f36`。

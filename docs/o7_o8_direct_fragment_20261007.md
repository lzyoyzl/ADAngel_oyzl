# O7/O8：直接供给寄存器的检查（v112）

结论：**新机制的单一候选未通过编译门槛，停止，不改变当前最佳版本。**
没有运行候选 GPU kernel，没有新的延迟、MSE或安全性结论。

## 先核对是否做过

按数据路径而非版本名称查重：

|已有测试|已验证的机制|与本次的区别|
|---|---|---|
|v85 预排 fragment|global → cp.async → shared → LDS → 寄存器|仍有 payload shared 中转和组间同步|
|v94 缩短 A fragment 生命周期|保留 shared，覆盖使用单个 A atom，减少逻辑 partial|仍有 LDSM、cp.async 和组间同步|
|v109 warp 私有流水线|每 warp 拥有 shared 输入与 cp.async buffer|仍经过 shared，且输入读取翻倍|
|v110 容量诊断|寄存器/shared 合成供数，真实 accumulator 数量|不是从真实全局输入读取并计算的 GEMM|
|本次 v112|预排 payload → `ld.global.ca.v4.b32` → MMA 寄存器|取消 payload shared 中转；全部 G128 scale 缓存仅发布一次|

本次复用 v85 修正后的 CuTe 排列和 v94 单 A atom 数学，不重新测试它们的原方案。
未扫描 tile、stage、链数、cache hint 或寄存器上限；仅一个 64×128×128、四 warp 候选。
这是组合改动，不是严格的单因素微基准，也不意味着省掉 barrier 就会更快。

## 方案与必须计入的成本

保持两个原生 INT4 路径、全部 32 个 G128 的独立 scale、整数安全 guard 和最终 FP32 输出。
源码包含原 FP32 scale 回退路径，但本轮未做 GPU 正确性验证，不能称为正式可运行后端。

目标为四 CTA 的寄存器预算，固定 `__launch_bounds__(128,4)`。
全 K 的 A/W scale 缓存需要 24,576 B dynamic shared，payload 不再经过 shared。
源码读取模型约为旧版的 **2.91 倍**，包括重复 payload 读取和 metadata；
这不是实际 DRAM 流量，也不是延迟预测，缓存命中率没有测量。

预排需要额外 24 MiB buffer，并增加 48 MiB 的在线重排读写。
若进入性能测试，W 重排必须计入 conversion-only/Cold，A 重排还必须计入 steady-state；
不能以免费的离线预处理替代。由于本次门槛失败，没有新转换或端到端成绩。

## A100 编译证据

CUDA 12.8、固定 CUTLASS commit、SM80；仅编译独立 cubin，不重编译正式扩展。

|指标：静态二进制，不是实测耗时|v78 控制|v112 候选|
|---|---:|---:|
|整数主循环指令|383|440（+14.88%）|
|分配 / 循环峰值活跃寄存器|168 / 166|128 / 126|
|循环原生 S4×S4 / U4×S4 MMA|32 / 32|32 / 32|
|循环 LDSM / LDGSTS / CTA barrier|16 / 10 / 1|0 / 0 / 0|
|循环 local load / store 指令|0 / 0|30 / 22|
|entry stack / spill store / spill load B|0 / 0 / 0|184 / 476 / 332|

128 个寄存器是编译约束下的结果，**不代表无代价减少寄存器或已实测四 CTA 驻留**。
减少 payload shared 操作后，仍增加了地址及 local-memory 工作；循环内有 52 条 local 读写，
同时未达到“循环工作减少”的预设门槛，因此不进入 GPU 性能测试。
spill 指令数不是实际字节流量；+14.88%也不能解读成实测变慢14.88%。
resource 报告的 `LOCAL:0` 不可当作没有 spill：编译日志、184 B stack 和循环 `LDL/STL`
共同说明实际机器码存在 local 路径。`SHARED:0` 是 static shared，另需上述 dynamic shared。

Host-only CuTe 检查通过：A 16,384、B 32,768 个 nibble 坐标，另能识别旧错误映射24,576处。
同一个候选 entry 保留双路原生 INT4，没有 INT8 MMA；旧 v78 控制机器码逐字不变。
这些只支持编译/坐标审计，不替代 GPU 输出、MSE、fallback 或 sanitizer 验证。

## 停止及保留

不放宽失败门槛，不扫描相邻寄存器预算、链数或直接供数变体，也不迁移到 O3。
O3 v89、O7/O8 v78+v73 主基准，以及既有 v99 微调的实测依据保持不变。
**没有新性能提升数字；当前目标尚未达到。** 原生扩展、production 默认、原始数据及5090不改。

实现提交 `d3537a05430a7efc8a823b42667eccd4550cdbcb` 先推 GitHub，再由 A100核验bundle/fetch/ff-only。
完整归档（本地及A100）为 `tmp/o378_v112_codegen_complete.tar.gz`，SHA256：
`b232f35a5a630c773be171b8ec16681f36ac67e791cdfc2fce106162c5bc2e59`。

- [源 SHA、命令、失败门槛、坐标和工作模型](evidence/a100_o378_roof_v112/reports/o378_roof_v112_direct_fragment_codegen/codegen.json)。
- [编译日志](evidence/a100_o378_roof_v112/reports/o378_roof_v112_direct_fragment_codegen/build.log)、[SASS](evidence/a100_o378_roof_v112/reports/o378_roof_v112_direct_fragment_codegen/o78_direct_fragment.sass)、[资源报告](evidence/a100_o378_roof_v112/reports/o378_roof_v112_direct_fragment_codegen/resources.txt)。
- [冻结证据说明](evidence/a100_o378_roof_v112/README.md)。

本地16项 CPU 契约/冻结证据及v111回归测试通过；它们重放源SHA、机器码、坐标报告和失败门槛，
不运行GPU，不能将其计为16项GPU正确性测试。

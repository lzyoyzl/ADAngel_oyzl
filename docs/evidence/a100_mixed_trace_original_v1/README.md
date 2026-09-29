# 原始 FP16 直接量化入口：合成验收

日期：2026-09-29；源码 `18332564cb9172f42350ee773fa89efe71d2b91b`。
代码先在本地修改并推送 GitHub，A100 fetch/fast-forward 后运行。
CUDA 无改动、未重新编译，沿用已审计二进制 SHA-256：
`f326244218ff8eefdfc27c63519e0e222fcdd110a64541afddeb70fb51fdbea8`。

这是**合成原始 FP16 fixture** 的运行入口验收，不是真实 24 样本结果。
本轮没有读取真实 trace 来量化，没有下载/传输原始 trace，没有覆盖旧 prepared。

| 检查 | 结果 | 证据范围 |
|---|---|---|
| 单元测试 | 33 passed | 格式、计时统计、NCU 汇总、输入策略/manifest 链接/路径安全 |
| 原始 FP16 fixture | 84 条四模式记录通过 | 随机 256³、全零 512³、随机 4096³，7 case×4 模式 |
| 公共准备重放 | 编码/scale/Q4 逐位一致 | fixture 在 CPU 从 FP16 准备，再由入口独立重放比对 |
| 源格式输入选择 | 原始 FP16 hash 一致 | 非零随机 W 的原始 hash 与 O0 反量化 W hash 不同，确认未误用 bridge |
| 输入错误拒绝 | 3 类通过 | FP32 冒充 FP16、错误 shape、与 prepared 不同的原始值 |
| O5/O6 vs 独立定点参考 | 最大绝对误差 0，MSE 0 | 两个布局、三个尺寸；不是相对 O0 的 MSE 为零 |
| 各模式/布局输出 | 逐位一致 | 固定输入的数值一致性，不要求各模式耗时相同 |
| 旧二次量化入口回归 | 56 条记录通过 | 随机 256³、全零 512³；没有更改其输入语义 |

运行参数 warmup=2、repeats=3、inner=10、rounds=1，仅用于接口与正确性。
不得把这些短测耗时当作正式性能，或用其证明真实 trace 的精度/性能目标。
这里的 manifest 单测使用 hash-only/mock fixture 检查边界；真实 raw 的深度
验证委托现有 `validate_raw_trace(deep=True)`。尚无真实 raw 可进行完整 CLI 验收。

## 证据

- `reports/mixed_trace_original_unit_v2.txt`：33 项测试；临时目录限定在 `/home/zlouyang/tmp`。
- `reports/mixed_trace_original_fixture_v1.txt`：原始路径运行完成日志。
- `runs/mixed_trace_original_fixture_v1/`：84 条原始计时、source hash、误差、环境/二进制身份。
- `reports/mixed_trace_secondary_regression_v3.txt`、`runs/mixed_trace_secondary_regression_v3/`：旧路径回归。

同二进制的 ISA/资源/内存安全证据见
[group-major 候选验收](../a100_mixed_group_major_v1/README.md)；本轮没有重新运行
SASS 或 Compute Sanitizer，不把 Python 入口测试宣称为新的硬件审计。

原始 FP16 与二次量化记录以不同 `input_policy` 区分，不能混合汇总。
正式数据仍待原始 trace 提供/传输，或用户明确选择二次量化备选。

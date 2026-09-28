# A100 O5/O6 真实 trace 入口的合成验收

2026-09-29。运行入口 `719c00d`；增加 4096³ 合成 fixture 的版本 `ce55951`。
没有修改 CUDA 或重新编译；沿用已审计的二进制 SHA-256：
`f326244218ff8eefdfc27c63519e0e222fcdd110a64541afddeb70fb51fdbea8`。

这不是正式 24 样本结果，也没有对真实数据执行源格式量化。
等待用户确认是否采用“已有 prepared → O0 FP16 操作数 → 在内存中进行
O5/O6 源格式量化”的二次量化口径，不能冒充原始 FP16 直接量化实验。

| 检查 | 结果 | 范围 |
|---|---|---|
| A100 现有 prepared 校验 | 24 文件集合、manifest、SHA-256 通过 | 只读文件与 hash，不作源格式量化 |
| 相关单元测试 | 30 passed | 编解码、计时统计、NCU 汇总、入口安全门与配对统计 |
| 小规模入口 fixture | 56 条记录通过 | 随机 256³、全零 512³；7 case×4 模式 |
| 含正式尺寸的合成 fixture | 84 条记录通过 | 以上两例加随机 4096³；不是 24 个真实样本 |
| 原生 O0 反量化 bridge | 与独立软件解码逐位一致 | FP16 A/W 各自检查 |
| O5/O6 对独立定点参考 | 最大绝对输出误差 0，输出 MSE 0 | 两个 scale 布局、三个尺寸 |
| 跨布局/四模式输出 | 逐位一致 | 不仅比较近似 MSE |
| 中间格式持久化 | 无 `.pt` 输出 | 只保留编码张量 hash、shape/dtype 与误差指标 |

入口测试使用 warmup=2、repeats=3、inner=10、rounds=1；仅作数值/接口验收，
不把这些短测的时间和 CV 当作正式性能证据。先生成合成 prepared，再使用
与正式入口相同的运行引擎；不会从真实数据目录偷取小样本冒充 fixture。

## 证据目录

- `reports/mixed_trace_input_check_v1.json`：实际 24 文件只读校验结果。
- `reports/mixed_trace_runner_unit_v1.txt`：30 项单元测试。
- `runs/mixed_trace_runner_fixture_v1/`：小规模 56 条记录。
- `runs/mixed_trace_runner_fixture_v2/`：含 4096³ 的 84 条记录，provenance、
  source-format hash、逐张量误差账本、输出误差、原始计时和 GPU 快照。
- `reports/mixed_trace_runner_fixture_v{1,2}.txt`：运行完成状态。

CUDA 没有变化，本轮没有重新跑 SASS/Compute Sanitizer，沿用
[同二进制审计与内存安全证据](../a100_mixed_group_major_v1/README.md)。
正式输入命令及统计口径见 [协议](../../o5_o6_a100_protocol.md)。

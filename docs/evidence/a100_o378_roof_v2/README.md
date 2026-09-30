# 第二轮片段交错与 NCU 证据

二进制编译源码 `5f96487`；各 run 的 `environment.json` 和审计 JSON 保存实际二进制 SHA-256。
正式默认不变，tune=-1 为原正式 symbol，tune=2 为独立 N atom 交错，tune=6 为扩大 N fragment 后交错。

顺序审查：旧 trace runner 对两个实现的轮换被随后反转抵消，不能当作平衡 AB/BA 证据。该版本正确性与独立 NCU 计数仍可用；需用修正后的 runner 重做正式性能，不能从现有微小差距直接选择默认。

- `runs/o378_roof_v2_screen`：192 个小矩阵/边界逐位检查，随后 4096³ 三轮合成输入性能筛选。
- `runs/o378_roof_v2_four_smoke`：1 个原始 FP16 trace，3 路径×3实现×4模式，共36条，非正式24样本性能验收。
- `reports/o378_roof_v2/audit`：18 个 candidate 实例均为两路原生 INT4＋cp.async；允许少量 spill，但不豁免 ISA 检查。
- `*_memcheck`、`*_synccheck`：各96个正确性检查，Compute Sanitizer 报告0错误。未进行 racecheck，不能据此宣称排除所有数据竞争。
- `reports/o378_roof_v2/ncu_*`：同一二进制各正式函数和 tune6 的 NCU 原始指标、SASS 与日志；没有丢弃慢记录。

NCU 单 kernel 诊断值（不是 CUDA Event 正式性能）：

| 路径 | 正式 / tune6 duration ms | 正式 / tune6 eligible warps每scheduler | 正式 / tune6发射活跃比例 |
|---|---:|---:|---:|
| O3 | 0.488832 / 0.465376 | 0.709 / 0.786 | 43.95% / 45.60% |
| O7 | 0.524928 / 0.490336 | 0.800 / 0.888 | 44.68% / 47.24% |
| O8 | 0.524960 / 0.488480 | 0.801 / 0.888 | 44.72% / 47.22% |

说明：NCU 改善支持独立链交错的方向，但收益有限；shared 服务工作量没有明显减少，math/MIO 等待仍然存在。没有锁频，诊断时间不能直接替代配对 Event 测试。`runs/*ncu*` 及 sanitizer run 中的 Event 时间受工具插桩/重放干扰，**禁止纳入普通性能汇总**。

原始 `.ncu-rep` 保留在 A100 项目的同名报告目录，此处仅同步轻量文本导出。所有中间/派生计数均以原指标口径解释，不把 theoretical sectors 误作 HBM 字节，也不将 stall 比例当作时间分解。

# v105：运行时因子相等的只读机会统计，未进入优化测试

48 条记录 = 24 个真实样本 × O7/O8。原始 FP16 activation 直接使用现有源格式量化器，
source identity 全字段与已测 v99 一致，保留 G128 scale、anchor/factor、row guard。
未执行新/旧 GEMM、未测新的 MSE/性能、未重新编译原生扩展、未改默认或5090。

- `reports/o378_roof_v105_runtime_factor_r2/`：完整结果、summary、环境与 source SHA、CuTe坐标检查；
  48 个 NPZ 保存原始 scale codes、group sum squares、base multiplier、factors、row status。
- `reports/o378_roof_v105_runtime_factor.log`：首次 host 坐标断言拦截的错误，尚无真实数据统计。
  修正统计脚本的 M32 行分组后通过，生产 kernel 从未改动；不把修复算作新优化。
- `tests/unit/test_roof_v105_evidence.py`：CPU 重建全部48个 factor/row guard、复算全部统计、
  核对 source identity 和代码 SHA、重放停止决策。NPZ 不是模型或 FP16 trace。

整 warp 四行向量相等的比例：O7 **0.441488%**、O8 **0%**。
“只做四向量相同复用”的乐观循环指令减少比例：**0.073221% / 0%**；
全部向量模式的理想复用也仅 **0.945370% / 0.000552%**。
均未扣分类、分支、读取、寄存器以及 W/CTA fallback；不是实测加速比或理论 kernel peak。
停止此路线，不编译候选或做性能重测。

下载归档 SHA-256：`3da15b7b54e5a222e130e6a3eb89da324fa5e7860de5d4eabe8d5f83df034cb5`。

# A100 O5–O10：24 个真实样本原始证据

主报告：[分组结果](../../o5_o10_a100_results.md)。所有数值来自本目录的原始 JSONL，未删除离群值。

- 源码/脚本：`3a97ce219ac75be9a4f7cb9d81c6117611cbfb3f`；CUDA 构建源码为 `4507c6e`。
- 二进制 SHA256：`3186f84b7c35c1933389a7ccafc092324d82a162af2eb5a9ddd20b8e42d0e3a0`。
- A100-PCIE-40GB，CUDA 12.8，PyTorch 2.7.1+cu128；未锁频、共享 GPU。
- 24 个原始 FP16 样本；4096³；1 轮、warmup=50、repeats=200、conversion inner=100。
- O7/O8：group-major scale；O9/O10：自然 scale；四者均 64×128×256。
- 命名版本 3；计时契约版本 2。O5/O6 是 FP16，不是历史旧称的双 INT4。
- 864 条记录，其中 576 条为 O5–O10，另 288 条是 O0/O1/O3 辅助参照。

`runs/a100_o5_o10_final_v3/`：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 全部每次 Event 原始值、各阶段统计、kernel 标识、MSE、CV 状态 |
| `summary.json` | 完整覆盖与正确性状态、配对汇总；不是严格 CV 全通过声明 |
| `validation.jsonl` | 独立参考、解码张量、Binary/INT4 逐位一致性 |
| `source_formats.jsonl` | 源格式量化误差与编码 hash |
| `source_provenance.jsonl` | raw/prepared 对照及 CPU 重放的逐位验证 |
| `config.json` / `environment.json` | 参数、代码/二进制/原始数据 hash |
| `raw_trace_manifest.json` / `data_manifest.json` | trace 来源；没有模型权重或完整矩阵 |
| `gpu_snapshots.jsonl` | 运行期间时钟、利用率、温度和其他 GPU 进程快照 |

`reports/` 保存原始 stdout 和系统环境查询。
数值校验全部通过；O9==O7、O10==O8 的 FP32 输出逐位相同。
部分记录 CV≥3%，尤其 FP16 steady-state 的单次分阶段计时；这些记录仍进入 median/mean 汇总。
一次快照观测到额外进程占用约 20,994 MiB，附近记录存在明显延迟上升；不据此断言所有波动原因均已确定。

正式运行结束后才合并 `1108e59` 并重编译候选；没有在本次运行中途改变源码或二进制。
后续 v4 编译/诊断与复测须使用独立目录，不能与此目录拼接为同一轮。

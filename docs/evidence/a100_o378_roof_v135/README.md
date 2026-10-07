# v135：冻结 DP2A 候选的首次 GPU 运行证据

结果：完整24样本三轮配对吞吐−6.48%，不采用，当前最佳及正式默认不变。
[简明报告](../../o3_dp2a_runtime_review_20261007.md)；[独立重算摘要](analysis.json)。

保留42份原始文本（1805292 bytes），SHA/大小见`index.json`：

- `runs/o378_roof_v135_full24/`：首轮旧转换kernel计数断言失败，没有正式性能记录。
- `runs/o378_roof_v135_full24_contract/`：新增参考输入维度错误，没有正式性能记录。
- `runs/o378_roof_v135_full24_checked/`：完整144条配对记录，原始Event、24输入SHA、GPU快照、资源、数值验收及原始summary。
- `reports/`：三次运行日志与最终源/扩展/cubin SHA、22项CPU检查。

复用原[v115二进制和PTX/SASS审计](../a100_o378_roof_v115/README.md)，没有重新编译GEMM或单路INT8路径。
原始静态gate仍失败；`dp2a_build.json`记录GPU执行前的固定小spill复核，不是性能验收放宽。
metadata pack使用旧cubin里的GPU kernel；独立CPU逐元素核对，额外成本属于W转换/Cold。
本轮仅正式测compute-only，未采新conversion/Cold/steady、NCU或sanitizer，不据GEMM推算端到端。

源码运行commit：`c47fd55d8e85ee20f12f6af89019de16f5cfd1bd`。
候选cubin SHA：`cf22985184630cb8c5011c6be06df0337e3934ab973574f0aa56423d4318d782`。
正式扩展SHA前后不变：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

本地和A100均保留完整归档（含运行host库等二进制；GEMM原cubin仍在v115目录/旧归档）：

```text
tmp/o378_v135_runtime_complete.tar.gz
SHA256 98a9d5ec6075d5ba08f494592e326d0b8e7ce3f2058fe50ebc001421baf79274
```

CPU复核，不重编译或运行GPU：

```bash
python -m pytest tests/unit/test_o3_dp2a_runtime.py tests/unit/test_roof_v135_evidence.py -q
```

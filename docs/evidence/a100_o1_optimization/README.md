# A100 O1 当前优化的验收证据

完整解释见[报告](../../a100_o1_optimization.md)。仅代表SM80/A100，不覆盖或替换5090结果。

- `paired_results.jsonl`：24样本×3实现×10轮，720条；每轮20次CUDA Event。
- `paired_summary.json`：全部原始计时汇总，配对bootstrap区间和所有CV异常。
- `four_modes_results.jsonl`：24样本×O0/production×4模式，192条；每阶段200次，转换inner100。
- `four_modes_summary.json`：四模式汇总；异常不丢弃。
- `*_environment.json`：源码commit、扩展SHA256、运行参数。
- `validation.json`：128项逐位输出比较和非法scale检查。
- `o1_o3_validation.json`：96项O1/O3语义正确性回归。
- `memcheck.txt`、`racecheck.txt`：包含K64/小尺寸生产分支的零错误验收。
- `unit_tests.txt`：60项单元测试。
- `audit.json`、`resources.txt`：实际function级PTX/SASS和资源审计，无local/stack/spill。
- `baseline_ncu_*`、`production_ncu_*`：不锁频NCU full报告的文本/原始指标导出。

最终实测二进制SHA256：
`1320b16d9a17439f0d3dd1a666ed92eec97b5b6070fec2c0af91a9c946f92f07`。
核心源码commit：`bfc8959050dcc9219ddfae4c17d607c570517bad`。
后续`7bc79f1`仅扩充Python验证的形状覆盖，没有改动CUDA实现。

完整`.ncu-rep`、PTX/SASS及构建日志已下载到本地`reports/a100_o1_opt/`，未提交Git。
服务器传输包的SHA256经本地复核一致：
`0b7d056147d5bc0abda0269080623197dc0aef56ae9688ba5b423dad9d535fe3`。
GPU进程快照只留在本地/服务器原始run目录，未把其他用户的进程路径提交Git。

这组证据支持“当前O1明显快于旧O1且MSE不变”，不支持“当前O1已快于O0”，
也不支持把含CV超标阶段的配对运行标为全量严格稳定通过。

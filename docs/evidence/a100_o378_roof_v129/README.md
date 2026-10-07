# v129 原始编译证据

源码先推送 GitHub，再经增量 bundle 在 A100 项目 fetch/ff-only 到
`182f79b29c3662475b1cfa1aab88ec8fbb191e79`。
CUDA 12.8.93 / pinned CUTLASS；原 v78 对照完整 SASS 编码不变。

候选同 entry 保留 64 条原生 INT4，增加 16 条 U8 MMA 计算系数外积。
IMAD 族158→115，但新增54条PRMT、热local读/写3/3，总静态循环383→461。
原工作量/local投入门槛失败，停止，没有候选 GPU 执行、性能、MSE、sanitizer 或 NCU。
CPU 穷举与 CuTe host 坐标通过不能当作 GPU 数值/安全验收；静态增加20.37%不是实测慢20.37%。

14 份原始文本（7,105,468 B）逐字节保留，SHA见 [index.json](index.json)。
完整压缩包 `tmp/o378_v129_verified.tar.gz` 包含 CUBIN 与 host 坐标验证程序；
本目录只保留文本，相应二进制 SHA 在 codegen.json。
压缩包 SHA：`5b13a76b2c5725979490bc0599f82d88f7165c665449fb9a2b4859fe5e3cd9d1`。
正式扩展 SHA 未变，当前最佳与 5090 保持。

复现（仓库相对路径；输出必须为新目录）：

```bash
python scripts/probe_o78_tensor_factor_codegen.py --output reports/o378_roof_v129_replay
python -m pytest tests/unit/test_tensor_factor_codegen.py tests/unit/test_roof_v129_evidence.py -q
```

编译器/审计器成功保存结果可退出0；是否进入 GPU 测试看 `codegen.json.cost_gate.passed`，
不能只看进程退出码。不放宽原门槛，不做相邻布局扫描，也不把该候选移植为正式默认。

# v96：八槽位 N64 tail pipeline 的编译证据

结论：编译、同 entry 原生两路 INT4/cg copy、旧对照编码审计通过，但排序 gate 均失败。
第二片段第一条 MMA 仍是每组第33条；已启动未完成链峰值仍8。停止候选，没有新 GPU 性能/MSE。
这些指标是静态 SASS 顺序，不是动态并发数、时间分解或加速测量。

源实现先提交并成功推送 GitHub：`54dbe9c1bc85f8b54aca446ab5d6a11cf9b8dfd5`，
再通过已验 SHA 的增量 bundle 在 A100 fetch + fast-forward merge。
CUDA12.8.93、SM80、固定 CUTLASS `db1c288993354c88e551c40c19a8fb93a774a241`。
实现只在本地 `/root/ADAngel_oyzl`；服务器文件写入仅在 `/home/zlouyang/ADAngel_oyzl`。

`reports/*_codegen/`：原始 codegen.json、生成头文件、PTX、SASS、liveness、resource usage、日志。
`reports/*_baseline_schedule.json`：本地对先前最佳真实 SASS 的重新追踪，不是新 GPU capture。
`reports/*_equivalence.json`：本地对服务器新 SASS 的完整编码比较，只有 symbol 文本被归一化。
旧对照与先前旧 binary 的比较通过；候选与旧 entry 编码不同，不能混淆这两种比较。
`tmp/*_compile.log`：原始服务器编译输出。

全部文本文件由原始归档机械复制，未改写。完整 cubin 原始归档不进入 Git：
`tmp/o378_v96_compile_complete.tar.gz`，两端 SHA256：
`d357cc74c3f57e061074dc957857e54a46db12d50c51082c6fc9bf45eabc3d16`。
正式 `_sm80.so` 校验值仍为：
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

本轮没有候选 kernel launch、Event计时、输出/MSE测试、NCU、sanitizer或CUDA占用查询。
CPU/source测试和编码检查不能替代GPU正确性；文档MSE仅引用已验收最佳版本，未冒充v96结果。
当前最佳、正式默认、5090均不变。

离线复核：

Git 中的离线测试验证源文件/文本哈希、旧对照编码、原生指令和数据流，不宣称重新验证
Git 中未保存的 cubin。运行驱动的完整 `checked()` 仍要求 cubin 存在且 SHA 一致，未放宽。

```bash
python -m pytest tests/unit/test_interleaved_tail_probe.py tests/unit/test_roof_v96_evidence.py -q
```

[本轮简报](../../o3_o7_o8_interleaved_tail_20261006.md)。

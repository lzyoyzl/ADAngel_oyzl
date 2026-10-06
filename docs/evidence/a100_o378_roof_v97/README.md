# v97：固定两条 startup 链的编译证据

结论：编译、同entry原生两路INT4/cg copy、先前最佳的完整编码对照通过；预设排序gate均失败。
真实SASS仍8条已启动未完成链，第9链仍在第33条MMA启动。没有候选GPU性能/MSE结果。
O7/O8热循环383→377条静态指令、活跃GPR166→164，但分配仍168；这不是实测提速。

源实现先本地提交并成功推送GitHub `8eeeb04e6e5e76ef3a000c10f826501e5bdaef4b`，
再通过SHA校验增量bundle在A100 fetch/fast-forward merge。
CUDA12.8.93、SM80、固定CUTLASS `db1c288993354c88e551c40c19a8fb93a774a241`。
服务器本轮所有写入均在 `/home/zlouyang/ADAngel_oyzl`。

`reports/*_codegen/`：原始codegen.json、生成头文件、PTX、SASS、liveness、资源与编译日志。
`reports/*_equivalence.json`：本地对下载的真实SASS作精确编码比较，仅归一化symbol文本。
先前最佳entry的编码对照通过，候选与旧entry的完整编码不同；不得混淆两种比较。
`tmp/*_compile.log` 和 `tmp/o378_v97_cpu_tests.log`：原始A100输出，后者124项CPU/source测试通过。

所有原始文本机械复制、未改写。完整cubin归档不进入Git：
`tmp/o378_v97_compile_complete.tar.gz`，两端SHA256：
`90dc034a457955cf486e3654ab201ab36b1a1f0628e8df39483fc26db1a3ebf4`。
该tar包含编译报告和两份compile log；CPU日志单独通过SCP机械复制，不冒充tar内文件。
正式 `_sm80.so` 仍为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

本轮没有candidate kernel launch、Event测量、输出/MSE、NCU、sanitizer或CUDA占用查询。
CPU/source/编译测试不代替GPU正确性；最佳、正式默认与5090均不变。
离线测试验证所有源和文本哈希、精确entry指令、SASS链数据流及负gate。
Git未保存cubin，不宣称离线测试重新验证了raw binary；运行驱动的完整checked()仍强制要求
cubin存在且SHA一致，没有因证据裁剪放宽。

```bash
python -m pytest tests/unit/test_tail_lookahead_probe.py tests/unit/test_roof_v97_evidence.py -q
```

[本轮简报与当前最佳MSE引用边界](../../o3_o7_o8_tail_lookahead_20261006.md)。

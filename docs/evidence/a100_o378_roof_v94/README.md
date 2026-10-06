# A100 v94：A operand 活跃范围的编译筛选证据

实现先在本地提交并推送GitHub，随后A100 fetch/ff-only merge。
源码commit：`9a6d4a3ecf9502c1fd4a39952fbc7cf1e419f4d6`。
Pinned CUDA12.8/CUTLASS，候选同entry原生U4/S4、S4/S4、异步copy和编码对照通过；
寄存器/CTA gate失败，没有GPU benchmark、MSE、NCU或sanitizer。

本目录保存不可变文本：codegen receipt、PTX/SASS、完整liveness和编译日志，
以及候选/对照的MMA链追踪。资源查询确认168regs、128threads、3CTA/SM，
仅加载CUfunction，没有执行目标kernel。

`codegen.json`的`partial_registers=16`、`independent_chains=4`、`logical_A_registers=4`
描述**源码意图**，不代表实际物理寄存器/在途数；SASS追踪峰值仍为8条链。
整个entry有8B stack/spill、回退循环有LDL，整数热循环没有local访问。
不使用`LOCAL:0`来否定已发现的spill，也不把编译spill字节当动态流量。

完整二进制及全部文本同时保存在本地和A100：

```text
tmp/o378_v94_complete2.tar.gz
SHA256=20e4e278bd920fd5ffce0792826db71f98ed1a95d14bf1e061d1d484f809e676
```

正式扩展SHA未变：

```text
94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462
```

复现到不存在的新目录，不覆盖原始证据：

```bash
python scripts/probe_operand_stream_codegen.py --output reports/new_v94_codegen
python -m pytest tests/unit/test_operand_stream_probe.py tests/unit/test_roof_v94_evidence.py -q
```

若gate未通过，内部benchmark驱动拒绝运行。当前最佳和MSE继续引用此前结果。
O3、5090、正式默认均未修改。

[简明结果与取舍](../../o3_o7_o8_operand_lifetime_20261006.md)。

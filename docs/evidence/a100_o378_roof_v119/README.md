# v119 原始编译证据

本轮不是性能测试。新方向仅取消两个纯寄存器 MMA asm 的 volatile，原输入输出约束、CuTe映射、
数学和pipeline不变。编译结果与v78完整编码SASS一致，投入gate失败，候选没有启动GPU。
没有新Event、MSE、转换、端到端、NCU或sanitizer结论；旧最佳/正式扩展/5090不改。

- 运行源码commit：`e92177f5cb8c65f7bfb9fcb38ab1b9f626c592e1`；先本地推送，再A100项目内同步。
- `index.json`：14份原始编译文本/生成header/环境记录，加一份上游BSD版权说明，各有SHA。
- 同entry PTX两路INT4/cg copy；同function SASS原生U4/S4与S4/S4、热循环0local。
- 全entry寄存器168/stack0；整数循环383条、活跃峰值166、64 MMA/16 LDSM/10 copy/1 barrier，全不变。
- 原始complete archive在本地和A100：`tmp/o378_v119_complete.tar.gz`。
  SHA-256：`003fe4c7d176a1a51f5f5df1cbc0720cb08fdd0f8960776e07bbdef3c8f4ad40`。
  含原CUBIN；binary未放入Git，不因为档案中的代码相同就宣称测得0%加速。
- 正式扩展SHA：`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，未重编译。
- 后续只修复未来生成副本的BSD注记保留，不改变本次原始记录、也不重新编译/重测同一优化。

```bash
python -m pytest tests/unit/test_register_mma_codegen.py tests/unit/test_roof_v119_evidence.py -q
```

CPU复算覆盖原文件SHA、Git源码commit、生成header、完整SASS、liveness、原控制binary及失败gate。
这些不是GPU数值/安全验收。[候选分析](../../o7_o8_register_mma_20261007.md)。

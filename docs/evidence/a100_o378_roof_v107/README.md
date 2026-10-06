# v107：固定预算 A/B 驻留轴交换，编译 gate 停止

仅一个新候选；没有重复 v48 N窗口、v49 B lookahead、v88遍历或v94 A短生命周期方案。
[结论及查重说明](../../o7_o8_residency_axis_20261007.md)。没有新增 GPU 性能、MSE或安全验收结果。

最终 `reports/o378_roof_v107_codegen_r4/codegen.json` 保存源码/产物 SHA256、命令、
CuTe坐标检查、同entry原生双INT4/cg copy、旧v78指令编码对照和完整 liveness。
整数循环383→379，分配168/活跃166均不变，MMA64/LDSM16/异步copy10不变，热循环local0。
预先固定的潜力 gate 为 false，不投入运行测试，不改变最佳/默认/正式扩展/5090。

失败日志依次保存：

- `o378_roof_v107_codegen/coordinates_build.log`：不同A/B tensor共用auto声明。
- `o378_roof_v107_codegen_r2/coordinates.log`：M64配置错误用于M32切片，被坐标断言拒绝。
- `o378_roof_v107_codegen_r3/build.log`：不同signed/unsigned slice共用auto声明。
- `o378_roof_v107_codegen_r4/`：修复后完整编译/审计成功，但性能潜力gate未通过。

构建 source commit：`e44142d9a4feddc247385c71a2cf8d8817a4fd80`。
候选 cubin SHA256：`bdd0e14bf90b6590e12e45ff397d879d0a1ac3636d571e8bd02f13fedbb465be`。
两端完整归档 `tmp/o378_v107_compile_complete.tar.gz` SHA256：
`b8831d57ba3fcf101893caff8a284530a4c64981e5326319440e77ab949f175f`。

Git只保存原始文本；cubin/坐标检查可执行文件保留于A100项目和本地完整归档。
失败日志不覆盖。CPU测试重算源码/文本SHA、SASS指令、liveness、旧编码对照及gate，
不把编译证明或host坐标检查冒充GPU数值正确性或性能。

```bash
python -m pytest tests/unit/test_residency_axis_codegen.py tests/unit/test_roof_v107_evidence.py -q
```

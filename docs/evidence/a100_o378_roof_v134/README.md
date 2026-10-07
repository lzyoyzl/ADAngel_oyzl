# v134：O3跨K加载地址缓存修复，编译门槛失败

`index.json`固定26份原始文本、10,549,764字节的SHA。
包含首次CPU嵌套shape断言失败目录、修正后的CPU验证、生成源码、PTX/SASS、liveness、编译日志与codegen门槛。
初次提交`2e7f327cef9322165db4510f3d947da814ab4576`，修正编译提交`9795c7d2bd68d8d6b495df1464e0d3028df5e34d`，均先推送GitHub、A100再fetch/ff-only同步。

原v89完整控制编码不变。3072源地址/12288目标word经原CuTe Copy_Atom调用核对；候选仍同entry两路原生INT4、8次下一A加载后的32次旧partial加权。
循环原候选374→354，但比最佳323仍多9.60%，新增4条热LDL；allocated168，无热STL/shuffle。
原最多+5%工作和热local≤1门槛失败，停止此修复。没有候选GPU、Event、MSE、NCU、sanitizer或实际驻留结果。
正式扩展/默认/5090不改，最佳不变；静态工作差不能当实测速度差。

完整归档（含CUBIN和host verifier）位于本地及A100项目内`tmp/o378_v134_compile_complete.tar.gz`，
SHA256=`f6b1b5b665c1ab04c47a8f6a599fe24fc5430cf0c0fa10ab1a084d655e3d0b72`。
二进制不提交Git。CPU证据复算测试：

```bash
python -m pytest tests/unit/test_o3_cached_a_iterator.py tests/unit/test_roof_v134_evidence.py tests/unit/test_o78_mma_phase_stalls.py -q
```

[结果与停止理由](../../o3_cached_a_iterator_20261007.md)。

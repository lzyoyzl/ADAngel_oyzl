# v126：O7 packed E4M3 转换编译证据

结论：相对已测 LUT，456→464 静态指令，31寄存器/8CTA驻留不变；未达到预设至少5%指令减少门槛，停止。**未执行候选GPU kernel，无新的Event、MSE、NCU或sanitizer结果。** 见[简明说明](../../o7_packed_mxfp8_conversion_20261007.md)。

- `reports/o378_roof_v126_codegen/`：第一次编译成功，审计因Mx8枚举编号写错而退出，没有build receipt；失败日志在`tmp/o378_v126_codegen.log`。
- `reports/o378_roof_v126_codegen_r2/`：修正定位脚本后完整编译和CUDA资源查询；`build.json`记录源文件/产物SHA、旧LUT和packed权重entry完整机器码比较，以及原样保留的失败gate。
- 两次生成的CUDA代码不变。修正没有改变算法、投入门槛或任何正式后端；增加基于旧真实SASS的定位回归。
- CUDA资源API只查询属性和理论最大驻留，不启动候选，也不是性能测试。CPU逐编码与packing测试不等同于GPU验证。

19份原始文本按字节保存，共4,692,686 bytes；`index.json`可逐文件核对。带`.so`的完整归档保留在本地及A100项目内：

```text
tmp/o378_v126_verified.tar.gz
SHA256 4ac4c385c618637537b1706245f328f90b6151a6d49a82cc2ef6f40d23e1c24a
```

第一次源码commit `058521dbb71ec12006de964943d0a64354a13696`；修正后 `436e974e355d3c3e85eea32603db925546ff8109`。均先在本地实现并推GitHub，再到A100项目内fetch/ff-only。
正式扩展SHA仍 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，当前最佳、正式默认和5090不改。

CPU重放，不进行GPU测试：

```bash
python -m pytest tests/unit/test_mx8_swar_codegen.py tests/unit/test_roof_v126_evidence.py -q
```

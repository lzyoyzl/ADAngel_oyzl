# v36：O3直接目标布局转换（待A100验收）

GEMM54、量化和G128 scale语义不变，仅内部`conversion_impl`选择：

|ID|路径|
|---|---|
|0|现有Q4转换、W scale转置、payload重排；激活split后重排|
|1|精确寄存器Q4 LUT、标量目标布局转换；同时转置W scale|
|2|同1，向量化输入输出；A读取8B、W读取4B，每线程处理8元素|

不改变正式默认、O7/O8或5090。每次转换完整包含查表、packing、payload和scale重排；
不把转换离线，计时区间无申请。向量路径明确验证指针对齐和索引范围。
诊断自然布局仅在计时后导出，GEMM直接使用最终G128-major数据。

验收：四模式、同GEMM54输出逐位一致、独立语义参考、MSE/O0、非默认stream、
转换编码与资源审计、原生双INT4审计、旧GEMM codegen比对、有限sanitizer。
真实24样本50/200/inner100配对轮换，不删除离群，CV全记录；未测前不预报收益。

```bash
python scripts/validate_o3_conversion_pipeline.py --output runs/o3_conversion_validation
python scripts/audit_o3_conversion_pipeline.py --output reports/o3_conversion_audit
python scripts/benchmark_o3_conversion_pipeline.py --output runs/o3_conversion_four24
```

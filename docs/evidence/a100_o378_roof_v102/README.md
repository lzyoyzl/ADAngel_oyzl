# v102 小INT4 atom容量诊断：原始证据

结论：相同物理操作量下，小atom的配对诊断容量 **0.765758×**，未达到预设1.10×投入门槛；
停止完整GEMM实现，不修改最佳或正式默认。不是O3/O7/O8真实样本性能/MSE的新结果。
完整解释见 [主报告](../../o3_o7_o8_small_atom_capacity_20261007.md)。

- `reports/o378_roof_v102_capacity/`：首次构建的SASS/PTX/资源日志；小atom255个SASS MMA，
  与256个PTX不符，审计失败，**没有GPU运行结果**。
- `reports/first_failure.log`：首次异常记录。源码commit `9e47ae1aa9ec193e1af95f15b090ab5887eac033`。
- `reports/o378_roof_v102_capacity_r2/codegen.json`：修正后的源码、命令、artifact hashes及审计。
  构建commit `fc078094afb963e4ba9b9ea0589a0a1de5192387`；同形seed避免重复起始计算，不放宽审计。
- 同目录 `results.jsonl`：2×200次原始CUDA Event和24项checksum状态，未过滤。
- 同目录 `summary.json`：可由 `probe_small_atom_capacity.summarize` 重算的配对结果。
- 同目录 `kernel.sass/kernel.ptx/resources.txt`：232/361条循环、原生IMMA.16864/IMMA.8832、49/47寄存器，
  shared padding由运行时设置50688B，CUDA查询3 CTA；不能仅看静态SHARED:0否定容量设置。
- `gpu_before.txt/gpu_after.txt`：前后1410MHz快照，不是逐launch锁频证据。
- `reports/fixed_run.log`：修正后的完整构建、审计及测量输出。

诊断不包含真实矩阵搬运、scale、全K输出accumulator和FP32 output。源码16条链不等同于实际同时在飞的16条MMA。
两种shape都观察全部D寄存器并有相同仿射C种子/checksum，**不能与v82绝对时间混用**。
两次构建目录均保留在服务器，完整二进制归档仅保存在项目tmp、不进Git：
`tmp/o378_v102_complete.tar.gz` SHA-256
`74e012072631005208070ac69a1c7a176d85e8c240f3beb69d394832bbe2e5df`。
正式A100扩展另核对SHA-256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

复核：

```bash
python -m pytest tests/unit/test_small_atom_capacity.py tests/unit/test_roof_v102_evidence.py -q
```

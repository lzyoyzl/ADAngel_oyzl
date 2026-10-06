# v100：O7/O8 exact signed radix16 编译筛查

结论：不晋级 GPU 测试。数学表示可行，但实际 SASS 没有减少循环工作或寄存器预算。
仅 CUDA 12.8.93、SM80、固定 CUTLASS commit 的编译结果，不是 GPU 正确性/性能/MSE 验收。

- 源码提交：`264cd712b77bd0f26aed767830535f60d33ad0a2`。
- CUTLASS：`db1c288993354c88e551c40c19a8fb93a774a241`。
- [原编译收据](reports/o378_roof_v100_codegen/codegen.json)绑定源文件及生成物 SHA-256。
- [复核分析](analysis.json)重放源编码穷举、旧完整机器编码、liveness、同 entry PTX/SASS。
- 原始归档：项目内 `tmp/o378_v100_compile_complete.tar.gz`，SHA-256
  `e93ff2576d31f5e9584aa4ea00d3ad710f843a5857f88c223d466a075553d050`。
  cubin 仅保存在该归档与原构建目录，不放入 Git。
- A100 正式 `_sm80.so` SHA-256 仍为
  `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

复核命令：

```bash
python scripts/analyze_balanced_digits_codegen.py \
  --input docs/evidence/a100_o378_roof_v100/reports/o378_roof_v100_codegen
python -m pytest tests/unit/test_o78_balanced_digits_probe.py \
  tests/unit/test_roof_v100_evidence.py -q
```

控制 entry 是 `adangel_roof_o78_eight_chain_candidate`；候选为
`adangel_roof_o78_balanced_digits_candidate`。候选需新的补偿高位，不能直接输入旧 packing。
没有在线 preparation 的 GPU 编译/计时，没有 candidate launch、NCU、sanitizer、24 样本输出或 MSE。
不能把 CPU 整数等价证明当作完整 GPU 验收，也不能把编译停止写成实测加速百分比。

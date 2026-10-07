# v124：O3代数转置MMA的编译审计

范围：CPU代数与CuTe坐标、A100编译/反汇编；**候选从未GPU运行，没有新性能、MSE或sanitizer结果**。

原始文本16份，路径/字节数/SHA见[index.json](index.json)。生成源、CUDA命令、PTX、SASS、nvdisasm活跃范围、resource和最初固定gate全部保留。二进制仅在项目内归档：

```text
tmp/o378_v124_complete.tar.gz
SHA256 63769181d81148e300042d3a60a7a1d295d810906fba8586b00b13d4d36f2c80
```

源码commit `cad527394266c7a4ee0fe1f7a91164ac23b33a38`。先GitHub推送，再A100 fetch/ff-only；默认、正式扩展和5090不改。

| 编译指标 | 原v89 | v124 |
|---|---:|---:|
| 热循环静态指令 | 323 | 322 |
| 活跃GPR峰值 | 166 | 160 |
| 分配GPR | 168 | 168 |
| 每线程独有W scale | 16 | 8 |
| factor load发射 | 8×LDS.64 | 8×LDS |
| 热local | 0 | 0 |

8192输出与49152操作数坐标通过；原控制完整编码一致，同entry原生双INT4/cg copy通过。MMA64/LDSM16/copy9/barrier1和普通IMAD65不变。未达预设3%指令或16GPR改善门槛，停止；不放宽gate、重测相邻参数或将编译差异当实测收益。

```bash
python -m pytest tests/unit/test_transposed_mma_codegen.py tests/unit/test_roof_v124_evidence.py -q
```

[完整说明、查重和停止依据](../../o3_transposed_mma_20261007.md)。

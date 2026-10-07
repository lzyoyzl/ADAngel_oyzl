# v121：factor-only read-only 供数的原始编译证据

只有编译和 CPU 回放，**候选从未运行**；没有新性能、MSE 或 GPU 安全结论。
源码 commit `2a36f6aec73bf9329d86caec854d141615decb1e` 已推 GitHub，A100 项目内 fetch/ff-only 后使用 pinned CUDA 12.8/CUTLASS 编译。

`index.json` 固定 13 份原始文本的字节数与 SHA，包含生成 header、PTX、SASS、liveness、资源、build log、源码/环境/正式扩展 identity。
CUBIN 未上传 Git；含二进制原始压缩包保留在本地与 A100 的 `tmp/o378_v121_complete.tar.gz`，SHA `00b3b71c93be8b17e856cb51992740acd4babbff7060adfc5d0377eabdafd56b`。

原 v78 控制完整编码相同，候选同 entry 仍为两路原生 INT4/16 LDSM/8 payload copy/1 barrier，168regs、零热 local。
但静态整数循环 383→534（+39.43%），未达到预先固定的至少 3% 减少门槛。停止，不扫描 cache-policy 邻居或迁移 O3。

三个残留 `LDS RZ,[RZ]` 不是 factor 数据读取；新的 factor 供数为 20 条 `LDG.E.CONSTANT`。
静态指令增加不是实测延迟增加百分比，不能报告“慢39.43%”或“提升0%”。

复算（不加载 CUDA、不启动 kernel）：

```bash
python -m pytest tests/unit/test_o78_readonly_factor.py \
  tests/unit/test_roof_v121_evidence.py -q
```

[瓶颈、改动及跨平台迁移说明](../../o3_o7_o8_bottleneck_portability_20261007.md)。

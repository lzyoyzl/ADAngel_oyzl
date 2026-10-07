# v131：同 G128 low-A 加载位置审计

本地源码先提交/推送 `212cde65ff6974e62fef956a29e04ba70dfccb48`，A100 在项目目录通过 Git bundle fetch/ff-only 同步相同 commit，再用固定 CUDA 12.8 / CUTLASS 编译。
bundle SHA：`d413813da681bf68ab3b854d451c22f3112b4bd26c862bbf9f521117f49df45f`。

15 份原始文本及 SHA 见 [index.json](index.json)，包括生成头文件、PTX、SASS、live-range、资源与完整编译日志。
原始归档 `tmp/o378_v131_verified.tar.gz` 包含 CUBIN；本目录不提交二进制。
归档 SHA：`675034764b3b49434e54b6325b9836dc84f0c2cdad48bb492b47b3e1ca048c40`。

结果：候选/控制完整 entry 的 2200 条指令机器字相同；每 G128 323 条静态指令、168 allocated registers、0 热 local、32+32 原生 INT4。
四条 low-A LDSM 仍位于 MMA 序号 0/0/12/13 之后，未达预设所有加载延到至少第8条 MMA 的门槛，因此停止，不继续调度表达式扫描。
这是“生成代码未改变”，**不是测得 0% 加速**；没有执行候选 GPU、MSE/Event/NCU 或安全测试。
正式扩展 SHA 不变，生产默认与 5090 不变。

回放：`python -m pytest tests/unit/test_o3_late_low_codegen.py tests/unit/test_roof_v131_evidence.py -q`。

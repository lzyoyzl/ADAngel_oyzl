# v130：直接 U32 系数 fragment 原始编译证据

源码 `6e504c82ba9316a0a7ab567f4eb1fd707457c6ec` 先推送 GitHub，
再在 A100 项目 fetch/ff-only 后使用 CUDA12.8.93 / pinned CUTLASS 编译。
这是对 v129 byte packing 的一次修正，未改布局/数学/guard/原投入门槛；旧失败证据保留。

PRMT 54→0、热循环461→406；但原v78是383，候选仍有热local读/写3/3。
IMAD族158→102，64条原生INT4加16条系数INT8；原对照完整SASS不变。
CuTe host逐坐标/byte检查通过，不代表候选GPU数值或安全通过。
原工作量/local门槛失败，停止该路线，无GPU执行、性能/MSE/NCU、默认或5090变更。

14份原始文本（7,140,506 B）原样保留，SHA见[index.json](index.json)。
完整压缩包`tmp/o378_v130_verified.tar.gz`保留CUBIN与host程序，本目录只跟踪文本。
压缩包SHA：`5a4f857a3e8185fb9eff88cd013eab30f5bcf85718837aa2647c25810b772cde`。
正式扩展SHA未变，见index；不把static增减冒充延迟或吞吐变化。

复现（仓库相对路径，新输出目录）：

```bash
python scripts/probe_o78_tensor_factor_words_codegen.py --output reports/o378_roof_v130_replay
python -m pytest tests/unit/test_tensor_factor_words_codegen.py tests/unit/test_roof_v130_evidence.py -q
```

编译/审计保存结果可退出0；是否投入GPU测试看`codegen.json.cost_gate.passed`。
本次为false，不继续packing/布局扫描、不放宽门槛。

# v127原始证据：全K激活factor shared面板

O7/O8完整24样本三轮配对吞吐分别**下降1.7076%/1.7186%**，不采用。
输出逐位相同、MSE不变，当前最佳、正式默认/扩展及5090不改。[简明报告](../../o7_o8_activation_factor_panel_20261007.md)。

- `reports/o378_roof_v127_codegen/`：完整PTX/SASS、资源、liveness、生成源码及编译gate；旧v78控制完整编码不变，两路原生INT4。
- `runs/o378_roof_v127_full24/`：288条原始记录，24样本×O7/O8×3轮×2实现；每条warmup1000/repeats200，inner100，compute-only，57,600次正式计时执行。保留全部CV与GPU快照。
- 旧/新CV≥3%：O7为4/72、2/72；O8为0/72、3/72。无过滤，不进行选优重测。
- 64合成、12边界、2个row/group factor专门检查；两个额外case实际使用整数快速路径，Af分别有6/48个不同值。
- `runs/*_preexecution_review.json`：原零热spill门槛保持失败；按用户允许少量spill，独立复核一次地址reload与实际3CTA后才执行GPU。每个进程启动候选前写入，不是事后修改gate。
- 有限candidate-entry memcheck/synccheck均0 errors，只覆盖small-M/N、完整K4096，不是4096³或racecheck全覆盖。
- `tmp/o378_v127_validate.log`保留首次JSON直方图键类型误报；在候选GPU执行前修正回放，不改CUDA或原gate。

28份原始文本10,246,120B按字节冻结，SHA和大小见`index.json`。完整归档在本地与A100项目中：

```text
tmp/o378_v127_verified.tar.gz
SHA256 aa98b269344e617f35f020a03d68ab23ec313238b769c96a60d7623449f5c5eb
```

CUBIN仅在完整归档，不加入此文本目录；其SHA在原codegen receipt中。
编译commit `f8ab39d61c1a92d7c26d0e1d7e65dc7d2098f99d`；运行commit `de07e9c28aba24809a85eb0a1050fbf15db78d02`。
正式扩展前后SHA均`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
本轮无新NCU/转换/Cold/steady计时；未将GEMM差值冒充端到端成绩。

CPU回放（不编译或执行GPU）：

```bash
python -m pytest tests/unit/test_o78_activation_panel.py tests/unit/test_roof_v127_evidence.py -q
```

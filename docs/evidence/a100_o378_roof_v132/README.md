# v132：O7 权重转换缓存系数表

源码先在本地实现并推送 `81f6412e8ec2dc15e99917deaa5f50377400a547`，A100 在项目目录通过 Git bundle fetch/ff-only 同步，再用固定 CUDA 12.8.93 / CUTLASS 编译。
bundle SHA：`8a7f5ad84742c8fbb5989cb6bb04ca1cfca2409015a3951406cca33dad887f44`。

11 份原始文本及 SHA 见 [index.json](index.json)，包含生成头文件、PTX、SASS、live-range、资源、完整编译日志。
原始归档 `tmp/o378_v132_verified.tar.gz` 包含 CUBIN；本目录不提交二进制。
归档 SHA：`8ee9296ba153e98f1b1b94a3eaad4e1758abf238d0897309835af75783df3c48`。

原 v78 控制完整 SASS 不变。候选保留同 entry 的32+32条原生 INT4、16 LDSM、1 CTA barrier，无热 local；168 allocated registers。
但查表使静态循环383→564（+47.26%）、IMAD族158→224（+41.77%）、LDS族15→71；供数增加至12条 async copy。
工作量与整数服务两个预设门槛失败，因此停止，不追加布局/表大小扫描，不运行候选GPU。
这不是实测“慢47.26%”：没有新增 Event、MSE、CV、NCU、安全性或实际驻留结果；建表 kernel 也未执行。
独立实验不接入正式后端，生产扩展SHA、默认、5090均不变。

回放：`python -m pytest tests/unit/test_o7_cached_coeff_codegen.py tests/unit/test_roof_v132_evidence.py -q`。

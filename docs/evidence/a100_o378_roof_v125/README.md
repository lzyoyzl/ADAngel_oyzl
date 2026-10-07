# v125 原始证据：O3高低路分别加权

结果：完整24样本、三轮配对吞吐**下降5.714%**，不采用，不迁移O7/O8。
当前最佳、正式默认/扩展和5090均不变。参见[简明报告](../../o3_split_weighted_20261007.md)。

- 编译：`reports/o378_roof_v125_codegen/`，原生双INT4、旧完整编码、全部SASS/PTX、live-range与32条两MMA链。
- 正式配对：`runs/o378_roof_v125_full24/`，144条记录、24样本×3轮×2实现；每条200次Event、warmup1000、inner100，compute-only。保留全部原值及双方各1/72 CV≥3%的记录。
- 准备、输入provenance、GPU快照、资源和96合成/8拒绝/2坐标检查随run保存；输出/payload/scale/status逐位一致，MSE vs O0不变。
- 有限安全：`runs/o378_roof_v125_memcheck/`、`runs/o378_roof_v125_synccheck/`及`tmp/`工具日志；同一候选entry、small M/N、K4096，均0 errors，不是4096³或racecheck全覆盖。
- 不隐藏初始失败：零热spill compile gate为false；依用户允许少量spill，GPU执行前明确追加资源复核（4条热LDL/0 STL、实际3 CTA/SM不降），两份决定均在build/environment中。不是把失败gate改写成通过。

55份原始文本按字节复制，SHA/大小见`index.json`。本地和A100项目内完整归档：

```text
tmp/o378_v125_verified.tar.gz
SHA256 cde6864eaecb1c728aa3f60c51a5472341f630bb47f233b3fd022d10b62163aa
```

CUBIN/.so保存在完整归档，不加入此文本目录。原始编译/runtime文件仍记录其SHA引用。
编译commit `99a13828f92d7d8cbca2c4be795295cf8c88f41a`；运行commit `d7ecb5727bd7c11e56df16124dd9e90930ac793b`。
正式扩展前后SHA仍`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
未新增NCU或正式转换/Cold/steady结果；不要从GEMM差值冒充端到端测量。

CPU回放，不编译或运行GPU：

```bash
python -m pytest tests/unit/test_split_weighted_codegen.py tests/unit/test_roof_v125_evidence.py -q
```

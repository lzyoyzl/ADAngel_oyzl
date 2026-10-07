# v115：精确 DP2A 后处理的范围与完整编译证据

新的标量 dot 融合机制，不是重复 tile、warp、stage、系数顺序或旧 low/high 普通重构。
候选源代码在本地修改，先成功推送 GitHub，再同步 A100。未修改正式默认、扩展或5090。

|阶段|实际完成内容|结论|
|---|---|---|
|全24样本范围检查|O3 UE8M0 scale，O7/O8原始 A factor 快照及 SHA|O3平均80.729% N128 tile可用；O7/O8仅条件性A侧上界，不能冒充完整覆盖率|
|最小 SM80 编译探针|PRMT + `IDP.2A.LO.S16.U8`|DP2A确实是单条原生指令；不是新的 Tensor Core 类型|
|完整固定 O3 kernel|两路原生 INT4、CTA64×128×128、4warp、3stage|整数热循环323→340，LDL 0→3；预先固定的成本门槛失败|
|CPU测试|整数范围/符号/packing、metadata、源码契约、编译门槛|14项通过，不是GPU正确性或MSE验收|

当前最佳仍是 O3 v89、O7/O8 v78+v73（v99 输出微调独立保留）。
没有候选GPU启动、新Event性能、MSE、conversion/端到端或sanitizer结论；不把编译淘汰称为实测负加速。

`reports/o378_roof_v115_dp2a_cost/` 保存72份全样本范围记录、ISA解析、PTX/SASS和日志。
`reports/o378_roof_v115_o3_dp2a_codegen/` 保存完整生成源码、PTX/SASS、live-range、resources、
命令、源SHA和成本门槛。旧v89控制编码相同，正式扩展前后SHA相同。
`reports/o378_v115_dp2a_unit.log` 为A100原始14项测试日志；`index.json`固定21份原始文本SHA。

标量采集源码commit=`7def7c9f584b4e47a4e0d1f32bab4ac791bf5f65`。
完整编译源码commit=`4ad0c9bef9e61561afad8294f44715a7afb2acef`。

完整归档（含两个 CUBIN）在本地 `tmp/o378_v115_dp2a_complete.tar.gz`，
SHA256=`bc963f5e6c7c42cf18e4a734949b960a6faa71993b436539f84550e7fad6b256`。
二进制保留在A100原reports路径及本地 `tmp/o378_v115_dp2a_extracted/reports/`，不提交Git。

```bash
# 仅重放已有文本/范围/历史源码的核验，不重新编译或启动GPU：
python -m pytest tests/unit/test_dp2a_recompose.py \
  tests/unit/test_o3_dp2a_codegen.py tests/unit/test_roof_v115_evidence.py -q
```

机制、查重与停止依据见 [v115说明](../../o3_o7_o8_dp2a_fusion_20261007.md)。

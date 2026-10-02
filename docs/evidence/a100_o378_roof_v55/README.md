# v55：O3 全K整数流式累加初筛

用户批准仅测试32个G128组成的全K候选，暂不修改默认。实现commit：
`6be5dba4839ca02351947999357f7f4f515fa5ee`。

## 做了什么

逐列选择原W指数最小值 `h`，每组仍以两路原生INT4计算 `P[g]`，
用INT32寄存器执行 `I += P[g] * 2^(e[g]-h)`，最后一次转FP32并恢复
`2^h`和原A行scale。没有重新量化、丢弃组scale或保留32组fragment。
CTA仍64×128×128、4个warp、3-stage cp.async.cg；只有一个累加器集合。

用Python大整数验证每列 `131072 * sum(2^(e[g]-h)) <= 2147483647`，
该绝对值界同时覆盖所有有符号乘积和累加前缀。仅K4096/正常UE8M0可进候选；
范围不安全或其他K回到旧控制kernel；code255报错。scale原地修改使guard缓存失效。

## 结果：没有提升，不升级默认

layer_00的q/k/v/o共4个真实样本，4096³，各3轮，warmup50/repeats200，
单stream、相同Driver CUDA Event计时、循环交换执行次序、不锁频、不删离群值。

|实现|Compute-only median ms|相对同轮控制的配对吞吐|CV≥3%的记录|
|---|---:|---:|---:|
|旧最佳O3/54控制|0.451584|1.0000×|5/12|
|全K候选v55|0.472576|0.9543×（−4.57%）|7/12|

配对加速比bootstrap95%区间 `[0.947817,0.986456]`；只有4个样本，
这是初筛证据，不是24样本正式结论。新旧输出逐位一致，MSE相对O0：
median **0.000278282719670**，mean **0.000283055340644**。
它们是本次4样本统计，不能替代既有24样本MSE。

36项GPU语义/回退测试通过（含全零、极值、行列组scale、非默认stream）；
2项同一tensor从安全到不安全的缓存失效检查通过，非法code255均拒绝。
本次未做完整sanitizer或24样本验收。

## 指令与资源

|项目|控制|v55|
|---|---:|---:|
|寄存器/线程|168|168|
|可驻留CTA/SM|3|3|
|动态shared bytes/CTA|50688|50688|
|ptxas spill stores / loads bytes|12 / 12|36 / 36|
|stack bytes|16|40|

同一kernel的SASS含 `IMMA.U4.S4`、`IMMA.S4.S4`、`LDGSTS.BYPASS`，无INT8 MMA。
控制入口与原扩展O3/54、O7/O8/59机器指令逐字相同；O7/O8哨兵未变。
整数方案移除了逐G128的I2F/FMA，但增加整数对齐工作，且spill增加。
最终缩放与输出写回交错还阻碍scale加载复用，故后续仅做一次针对性写回修正v55b。
不能仅据静态opcode数量判定动态瓶颈占比，本轮未用NCU给出占比。

范围检查和锚点构造是本次探针的CPU准备操作，每样本约27–33ms，独立记录在
`guard.preparation_wall_ms`，排除在compute-only之外。没有集成conversion、Cold或steady，
**不报告这些模式的收益，也不把准备成本视为免费**。

## 原始证据与复现

- `reports/o378_roof_v55/`：PTX、SASS、资源/编译日志、audit/codegen JSON、GPU preflight。
- `runs/o378_roof_v55_screen/`：环境/二进制SHA、全部CUDA Event原始值、逐样本MSE、GPU快照和summary。
- 原扩展SHA保持 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
- 下载归档SHA256：`d31c1c337ef33d88c206882985a14c10960ce83300c9cb0377d103fe83172b80`。
- 未提交cubin/so，只保存可复核的文本和数据。后续修正在独立目录v55b，未覆盖本轮。

在实现commit下，使用新目录执行：

```bash
python scripts/probe_roof_fullk_integer_codegen.py --output reports/o378_roof_v55
python scripts/audit_roof_fullk_integer_probe.py --directory reports/o378_roof_v55 \
  --best-sass reports/o378_roof_v54/after.sass > reports/o378_roof_v55/audit.json
python scripts/validate_roof_fullk_integer_probe.py --cubins reports/o378_roof_v55 \
  --output reports/o378_roof_v55/preflight
python scripts/benchmark_roof_fullk_integer_probe.py --cubins reports/o378_roof_v55 \
  --output runs/o378_roof_v55_screen --samples 4 --rounds 3 --warmup 50 --repeats 200
```

# v39：显式移位重组 partial，未确认收益

**结论：不采用。O3 GEMM 最佳仍为54，O7/O8仍为59；转换最佳、正式默认和RTX5090代码不变。**

## 测试的改动

保持两路原生INT4、CTA64×128×128、4warp、G128独立scale及FP32累加顺序，
仅把活动路径的 `partial = low + 16 * high` 改写为显式PTX
`shl.b32 + add.s32`。对有界INT32 partial是精确重组，不是magic-bias或近似转换。
O3三阶段、O7/O8两阶段cp.async.cg流水线不变。

编译三种隔离cubin：0为原C++表达式；1为同一asm块；2为分开的asm块。
私有device-body副本仅有namespace和这一条表达式的差异，CPU契约测试逐文本核对。
没有链接进正式扩展，也没有新增正式实验调度。

## 指令和资源审计

|项目|O3控制 / 候选|O7/O8控制 / 候选|
|---|---|---|
|总静态SASS指令|944 / 944|880 / 880|
|静态IMMA / I2F|64 / 64，均不变|64 / 64，均不变|
|寄存器/线程|168 / 168|168 / 168|
|stack|16B / 16B|8B / 8B|
|ptxas spill stores / loads|12B / 12B，均不变|8B / 8B，均不变|
|SASS文本变化条数|88|6|

控制0的完整编码SASS与当前最佳54/59完全相同。候选仍生成IMAD/LEA，
没有强制出更少的指令；全部opcode计数与控制相同。
文本差异主要是寄存器分配及可交换乘法操作数的次序，不意味着数学工作减少。
候选1和2的编码SASS完全相同，因此只对0和1做性能比较，避免重复测同一个实现。
所有六个entry均有U4×S4与S4×S4 IMMA及原cg异步搬运，不含INT8替代。

资源中的动态shared-memory由启动参数提供，不能把cuobjdump的SHARED:0解释成没有shared-memory。
同理，LOCAL:0不等于零spill；这里以ptxas日志与SASS为准。
opcode计数以修正谓词解析后的 `audit_portable.json` 为准；
首次 `codegen.json` 保留原始编译记录，不用于opcode分类结论。

## 24个真实样本性能与MSE

24样本×3轮，warmup50/repeats200；统一原生Driver CUDA Event计时，
同样本同轮循环换序，共432条未过滤的compute-only记录。

|后端|同轮控制 median ms|候选 median ms|配对吞吐变化|speedup描述性95%区间|
|---|---:|---:|---:|---|
|O3|0.476160|0.475136|0.00%|[0.996552,1.004320]|
|O7|0.500736|0.500736|+0.10%|[1.000000,1.003084]|
|O8|0.503296|0.502016|+0.15%|[0.998984,1.006094]|

速度比先在同一样本内汇总轮次，再跨样本汇总，不能直接用表中两列median相除。
三个区间均包含1，**没有确认提升**，也不从本轮较小的绝对延迟跨run挑选新最佳。
CV≥3%的控制/候选记录分别为37/35、36/36、35/34，分母各72；全部保留。
共享、未锁频GPU，没有根据离群值筛选有利结果；这些区间仅是当前trace的描述性比较。

|后端 / 配对FP16参考|Median输出MSE|Mean输出MSE|相对当前最佳|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位相同|
|O7 / O5|0.005536172426666|0.005053635833762|逐位相同|
|O8 / O6|0.004411084948645|0.004381379215302|逐位相同|

所有记录的被检查输出均为finite FP32且逐位等于54/59。原始FP16 trace及prepared文件SHA、
公共量化重放和O7/O8源格式provenance均保留；没有用二次量化代替原始输入。
本轮没有修改conversion，负结果不晋级四模式复测，不声称端到端改善。

## 检查范围和结论

- 180项GPU核心检查：3后端×5种shape×4种pattern×3种policy。
  K覆盖128/256/384/640/4096；包含随机、零、饱和、零scale及非默认stream。
  与当前最佳逐位相同，并通过FP64语义参考容差检查。
- memcheck、synccheck各在一个真实4096³样本、三个后端、两个不同cubin上检查，
  各6条记录、0 errors。仅是有限安全检查，不冒充穷举所有shape。
- A100相关CPU单元测试252项通过；归档后另有3项证据重算测试。
- 正式扩展SHA-256仍为
  `fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`，没有重新构建或覆盖。

这次实验否定的是“仅换成显式移位表达式就能降低整数重组成本”这一具体假设，
不证明所有依赖调度都无优化空间。MMA/I2F/scale工作没有减少，也没有确认缩小
与约0.220347ms固定工作理想容量下界的差距；该下界不是承诺可达到的延迟。
本轮未新增NCU采集，不把v37的stall比例当作本轮候选的实测数据。
后续不继续枚举这类等价表达式，需针对供数、复用或真实依赖图提出可在SASS验证的改变。

## 复现与证据

编译源码：`296320b4a896b9a9a405bf6a9bf88bdef32ba8ab`；
GPU预检：`89812bfa72e3fd38f12dee2f6846d5f64494715a`；
配对测量及最终审计：`72accc35b0cae7c997fb9e1396abbb692d545d91`。
全部先在本地实现、推送GitHub，再在A100项目中fetch/fast-forward；未修改项目范围外配置。

```bash
python scripts/probe_roof_recompose_codegen.py --output reports/recompose_recheck
python scripts/audit_roof_recompose_probe.py --directory reports/recompose_recheck \
  --best-sass reports/o378_roof_v37/audit/extension.sass \
  > reports/recompose_recheck/audit.json
python scripts/validate_roof_recompose_probe.py --cubins reports/recompose_recheck \
  --output runs/recompose_validation
python scripts/benchmark_roof_recompose_probe.py --cubins reports/recompose_recheck \
  --output runs/recompose_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

归档包括42个文本证据，SASS可重新审计，JSONL包含原始Event采样和MSE。
cubin/PTX/Driver .so留在A100项目对应目录，不提交二进制。
传输归档SHA-256：`f38f71e15e146aafba1c321c746c74332e1119f51e3bf96f6f1f650632a9b2b8`。

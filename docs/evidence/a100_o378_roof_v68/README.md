# v68：O7/O8 全 K 整数累加的 GPU 在线准备与四模式初筛

独立 GPU 准备已实现，数值与安全检查通过；但转换成本抵消 cached GEMM 收益。
四个真实样本中，O7/O8 Cold 配对吞吐分别 **−5.50% / −5.53%**，
steady-state 为 **−0.63% / −1.37%**。停止扩大这一版独立准备方案，正式默认不切换。

## 实现和费用归属

GEMM 直接复用 v67 已审计的 control/candidate cubin：CTA64×128×128、四 warp、
两级 cp.async、两路原生 U4×S4/S4×S4、独立 G128 scale 与 FP32 输出。
本轮没有修改 GEMM 指令或源量化，新增 GPU 因子、范数与 CTA 判定路径：

1. 原转换5生成 G128-major INT4 planes 和旧 FP32 scale。
2. 每行一个 CTA 从源 scale 编码生成精确整数 factor/base，同时重新读取 packed payload，
   计算加权平方和。UINT64 乘法高位与饱和加法防止 guard 自身溢出。
3. 每个 GEMM CTA 检查 factor 乘积、Cauchy 范数界，以及最终两次 FP32 乘法的保守范围。
   安全走全 K 整数累加，范围不满足走旧浮点路径，非法编码拒绝进入 GEMM。

范数饱和上限为 `(INT32_MAX)^2+1`，比较用整数除法，避免直接相乘溢出。
锚点涵盖全部非零源 scale，包括 payload 为零的组；量化数据与每组 scale 不改变。
GPU metadata 与 CPU 任意精度参考逐项比较。与 v67 相比，另对极端 FP32 base 增加保守回退。

全部 device buffer、Event 和 tensor multiplier 的读取在计时前完成；计时内无 allocation。
CPU oracle 仅用于验证，不参与运行路径或计时。没有 CPU 判定回传/同步再选择 GEMM。

|模式|候选计时范围|
|---|---|
|Conversion-only|W 转换+W metadata；A 转换+A metadata+CTA guard，各批量100次再摊销|
|Compute-only|已缓存上述结果，只计 GEMM|
|Cold|单次 W转换/metadata+A转换/metadata/guard+GEMM|
|Steady-state|缓存W，单次 A转换/metadata/guard+GEMM|

主路径直接计时先执行，之后单独统计摊销转换；conversion-only total 是每对阶段样本之和。
Cold/steady total 直接 Event 测量，不用阶段 median 相加代替。Compute-only total 等于 GEMM。

## 实测结果

A100，4096³，`layer_00_{q,k,v,o}_proj`，一轮、warmup50、repeats200、inner100。
同进程、单 stream、交错 control/candidate 顺序；64条记录全部保留。
控制为原最佳 tune59+conversion5，候选为 v67 整数 GEMM+本轮在线准备。
“变化”是逐样本配对吞吐变化，与两个汇总 median 的比值可能不同。

|后端|模式|控制 ms|候选 ms|配对吞吐变化|speedup 95% CI|
|---|---|---:|---:|---:|---|
|O7|Conversion-only|0.047017|0.112707|−58.28%|[0.415324,0.417647]|
|O7|Compute-only|0.493056|0.458240|+7.47%|[0.965675,1.085011]|
|O7|Cold|0.554496|0.585984|−5.50%|[0.940648,0.955768]|
|O7|Steady-state|0.531456|0.536320|−0.63%|[0.984934,0.996226]|
|O8|Conversion-only|0.070979|0.137510|−48.38%|[0.514633,0.520021]|
|O8|Compute-only|0.492544|0.458752|+6.77%|[1.065367,1.084821]|
|O8|Cold|0.568832|0.602112|−5.53%|[0.940767,0.954469]|
|O8|Steady-state|0.536320|0.543744|−1.37%|[0.974170,0.992381]|

O7 本轮 compute 区间跨1，不能单凭四样本初筛声称确认加速；此前 v67 的24样本×3轮
cached 证据仍单独保留。不同run不拼接最佳延迟或累计百分比。

|后端|转换阶段|控制 ms|候选 ms|新增约 µs|
|---|---|---:|---:|---:|
|O7|W|0.015252|0.042563|27.31|
|O7|A+所需guard|0.031764|0.070154|38.39|
|O8|W|0.030218|0.057434|27.22|
|O8|A+所需guard|0.040750|0.080108|39.36|

独立准备重新读取 A 约16 MiB、W约8 MiB，并增加两个 operand kernel 和一个 guard kernel。
新增转换成本约66µs，大于本轮 cached GEMM 节省的约34µs；这是端到端退步的直接证据。
以上差值包含 metadata 生成和额外启动，不等于单独测得的纯带宽或纯指令瓶颈。
本轮没有新增 NCU，不把延迟差定量归因到某一种 stall。

## MSE、资源与验证范围

|参考|实现|Median 输出MSE|Mean 输出MSE|
|---|---|---:|---:|
|O7/O5|控制|0.000102067943873|0.000099167494159|
|O7/O5|候选|0.000102067943158|0.000099167492985|
|O8/O6|控制|0.000123493256717|0.000122317650852|
|O8/O6|候选|0.000123493261233|0.000122317652713|

64条结果全部 finite FP32、MSE回归通过。控制与旧 tune59 逐位一致；候选允许 FP32
重新结合，O7/O8 相对旧输出 max abs 分别1.430511e−6/1.907349e−6，最大输出差MSE
3.249305e−15/3.405064e−15。这是首层四样本，不能与24样本MSE大小直接比较成精度提升。

- 46项 CPU metadata/计时契约及整数界测试通过。
- 64项 GPU 计算/四模式检查，覆盖随机、全零、正负交替、宽scale、非默认stream及回退。
- 12项 GPU 边界检查：源零码、UINT64范数溢出、饱和范数配零操作数、factor超界、
  非法 UE8M0/E4M3/E6M2、次正规 multiplier、epilogue中间/最终乘法超界。
- memcheck、synccheck 各运行以上小 M/N、完整K4096验证，均0 errors。
  未做本轮 racecheck，也不把有限 sanitizer 范围说成任意4096³安全证明。
- 三个 metadata kernel 各25 registers，guard30 registers，均零local/stack/spill。
  GEMM仍为168 registers/thread、8 B spill stores/loads，3 CTA/SM；零spill只指准备kernel。
- GPU metadata 与精确 oracle 一致，转换后的 payload/FP32 scale 与旧转换5逐位核对。
  GEMM沿用 v67同entry U4/S4原生INT4与LDGSTS审计，cubin SHA由harness校验。

未锁频、共享GPU；按 conversion/compute/cold/steady，选定阶段CV≥3%条数（各4条）：
O7控制 `0/3/3/4`、候选 `0/4/1/4`；O8控制 `1/4/2/4`、候选 `0/4/0/3`。
全部离群与CV失败保留，不声称通过严格全阶段CV<3%，区间仅是四样本的描述性统计。

## 结论与复核

当前应保留 v67 为 cached GEMM 候选、原转换5为完整路径对照。
下一项有针对性的工作是将加权范围统计所需的组平方和融合进转换，减少重复payload读取；
再用小 metadata reduction 生成factor/base/guard。只有这笔成本降下来，才值得扩大端到端测试。
这是待验证方向，不承诺融合必然提速。本轮不进行新的tile/寄存器参数扫描。

源码在本地编辑并推送GitHub，再经A100 fetch/ff-only merge，测量commit `d5dc5ae`。
正式扩展未重编译，SHA256仍为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
完整下载归档SHA256：`b97ffe50c4f1772b2ee9a1a1475f9c71d3290541cf5a1e0388e26165f19f11fd`。

[原始计时与汇总](runs/o378_roof_v68_screen/summary.json)、
[编译与源码校验](reports/o378_roof_v68_codegen_checked/build.json)、
[GPU验证](reports/o378_roof_v68_validation/validation.json)均随报告保存。

```bash
python scripts/probe_o78_gpu_prepare_codegen.py --output reports/v68_rebuild
python scripts/benchmark_o78_fullk_gpu_prepare.py \
  --gpu-build reports/v68_rebuild --gemm-cubins reports/o378_roof_v67_codegen \
  --output runs/v68_recheck --samples 4 --rounds 1 \
  --warmup 50 --repeats 200 --inner 100
```

输出目录须不存在；v67 cubin可用 `scripts/probe_o78_fullk_codegen.py` 重建，
其receipt绑定源文件，禁止混入未经核对的二进制。

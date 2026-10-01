# v47：四个 warp 的 M/N 分工消融

状态：A100编译、审计、数值/安全、24样本性能与NCU均已完成，两个候选均退化，不采用。
原54/59、正式默认和5090均不变。[完整结果](evidence/a100_o378_roof_v47/README.md)

## 问题与变量

当前最佳54/59的 CTA 为64×128×128，四个 warp 按2×2排列。
v40曾增加到八个 warp，虽消除spill却增加shared供数并变慢。
本轮保持128线程、64个FP32 accumulator/线程与原流水线，只改变四个warp的分工。

|候选|warp M×N|每warp输出tile|每个N流式片宽度|逻辑fragment供数字节/CTA/G128|
|---|---|---|---:|---:|
|p0：原54/59对照|2×2|32×64|64|32768|
|p1|1×4|64×32|128|40960|
|p2|4×1|16×128|32|40960|

逻辑供数为`2*64*64*WN + 128*64*WM`：前者是low/high A，后者是W。
它仅表示warp间重复读取的packed payload量，不是实测LDSM数、bank conflict或耗时。
两候选均多25%逻辑供数，潜在收益只能来自fragment寄存器生存期/调度的改善，不能预设更快。

## 保持不变与验证

- 原两路U4×S4/S4×S4、G128顺序与FP32运算；不进行magic或跨G128整数累加。
- O3三阶段、O7/O8两阶段cp.async.cg，原shared swizzle、scale布局和最终store。
- 修改仅为CuTe warp几何；CuTe identity坐标检查全部8192输出唯一归属、low/high坐标一致、shared布局与大小不变。
- p0完整编码SASS必须与54/59相同；所有候选同一正式probe entry检查原生INT4与异步搬运。
- 180项有限数值检查及memcheck/synccheck/racecheck；K128/256/384/640/4096，非默认stream。
- 24真实样本×3轮，50预热/200测量；候选循环换序，完整保留CV失败和离群值。
- 对当前最佳要求逐位相同，并重算O3/O0、O7/O5、O8/O6的FP64-reduced MSE。
- NCU仅解释指令、shared、spill、warp调度变化；不能替代CUDA Event性能。
- 初筛正向者独立确认；无确认收益则保留54/59，不扩大成无收益的正式四模式实验。

目标仍是缩小与资源容量上界的差距；0.220347ms是理想重叠必要下界，并非已证明可实现的延迟。

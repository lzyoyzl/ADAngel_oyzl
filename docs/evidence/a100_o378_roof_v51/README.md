# v51：O3 共享两组对齐参数，仍未超过原最佳

**不采用。O3=54、O7/O8=59仍是GEMM最佳；正式默认和5090版本均未修改。**
v51只测试已获准的相邻两个G128，没有增加合并窗口，也没有将此方法套用到O7/O8。

## 1. 本轮实现

保留每组原始UE8M0 scale，以较小指数为锚点，在INT32中精确重建两组之和，
再做一次INT32→FP32和FMA；不重新量化，不用近似公共scale代替原scale。

与v50相比，v51不再为每个输出元素重复计算指数差、检查范围和处理回退：

- 准备阶段检查整个W scale张量：相邻组指数差均≤13时，进入无逐输出判断的快速kernel；否则运行原v50通用回退kernel。
- 每CTA每列每对group只生成一次`factor0/factor1/anchor_scale`，存入shared供各输出行复用。
- 输出后处理为`sum=P0*factor0+P1*factor1`，随后`acc=fma(float(sum),anchor_scale,acc)`。
- 保留两路原生U4×S4/S4×S4、register partial、FP32输出、原始A row scale和量化语义。

`|P|≤131072`，指数差≤13保证合并绝对值≤1073872896，不超过INT32范围。
使用有界整数乘法，不对负的有符号数做C++左移。奇数组尾部保留处理，code255拒绝。
范围缓存以张量地址、版本和shape为键，并保留张量引用；原地修改会重新检查。

CTA输出仍64×128、4warp、每组K128，成对处理K256；四个payload槽和两组寄存器fragment仍在。
shared由v50的67584B增到68608B。该方案不属于仅修改求和顺序的单因素实验。

## 2. 24样本性能与MSE

24样本×3轮×两方案，共144条；预热50次、单stream CUDA Event测200次。
同进程循环换序，控制kernel编码SASS与原最佳54完全一致。

|O3方案|Compute-only中位延迟 ms|配对吞吐变化|speedup 95% CI|CV≥3%记录|
|---|---:|---:|---|---:|
|原最佳54，同轮控制|0.474368|基准|[1,1]|36/72|
|v51共享对齐参数|0.584704|**−19.30%**|[0.803136,0.810480]|21/72|

比值由同样本同轮配对汇总，不直接相除两列总体median。共享、未锁频GPU，
全部离群和CV失败记录保留，不宣称通过全部CV验收。控制记录的median范围
0.440320–0.482304ms，候选0.555008–0.594944ms，完全分离，退化量级大于本轮波动。
bootstrap仅是当前24样本的描述性区间，不排除系统噪声，也不能确定每个离群值的成因。

|O3方案 / O0参考|Median输出MSE|Mean输出MSE|与原最佳输出的MSE|
|---|---:|---:|---:|
|同轮控制|0.006653010287410|0.007578847013303|0|
|v51|0.006653010287410|0.007578847013303|0|

由FP64 reduction重新计算。全部24样本输出恰好逐位相同，但不对任意输入承诺逐位一致。
真实trace的最大相邻指数差为5，全部实际执行快速kernel，没有用回退结果冒充快速路径性能。

**范围检查不免费：** 此原型使用PyTorch准备阶段检查，一次性host wall time
min/median/max为0.540090/0.763392/12.728968ms，已逐样本记录在`guard`中，未计入GEMM Event。
这不是GPU转换阶段延迟，也不能拿来直接相加构造Cold时间；若晋级正式实现，必须在完整
转换/Cold流程中重新计量并优化。v51纯GEMM已退化，因此不晋级五轮确认和四模式测试。
本轮没有新增conversion-only、Cold或Steady-state结果。

## 3. NCU解释：I2F减半，但寄存器、供数与发射代价上升

首样本分别抓取一个正式probe launch，`--set full --cache-control all --clock-control none`。
核对同entry原生INT4、symbol、资源、静态opcode指纹和动态数学工作量。
NCU duration不是上表的Event时间，不能将两种计时混合计算加速比。

|指标|原54控制|v51|
|---|---:|---:|
|NCU duration ms|0.396992|0.510688|
|动态warp指令 M|99.320|114.352|
|IMMA M|16.777|16.777|
|I2F / FFMA，各 M|16.777|8.389|
|IMAD M|12.493|24.158|
|Shared wavefronts M|29.622|31.850|
|Local理论sectors M|3.178|16.712|
|Eligible warps / scheduler|0.625737|0.475356|
|Issue active %|42.445|37.635|
|寄存器 / 线程|168|255|
|编译器spill store/load bytes|12/12|120/120|
|Shared bytes / CTA|50688|68608|
|最大驻留CTA / 计算warp，每SM|3 / 12|2 / 8|

LDSM均4,194,304，MMA及片段读取的总工作没有减少。v50逐输出的BSSY/BSYNC已消失，
v51实现了减少分支工作的设计目的；但整数重建、参数读取和spill使总动态指令仍比54多15.14%。
local理论sector约5.26倍，驻留warp减少，eligible和issue均下降。
这是寄存器活跃范围、供数和调度方面的证据，不意味着能把退化精确归因于其中一个计数。
shared excessive wavefront为0，不应误判成bank conflict；sector不是实测HBM字节。
stall采样仅用于定位等待，不是耗时分解。

在108SM、1410MHz、理想资源重叠的必要服务时间模型下：

|资源下界 ms|原54|v51|
|---|---:|---:|
|MMA|0.220347|0.220347|
|I2F|0.220347|0.110173|
|全部指令发射|0.163055|0.187733|
|L1TEX数据wavefront容量|0.177682|0.219460|
|模型最大值|0.220347|0.220347|

v51移除了I2F与MMA并列的容量约束，却增加接近MMA下界的L1TEX工作。
**理想下界不变，实际更慢，说明减少I2F不足以保证接近上界。**
这些数是必要容量条件，不是可达到延迟承诺；不把运行频率不同的Event时间解释成精确效率。

后续优先设计较短的寄存器生命周期：尽量复用两组A/B fragment的存储，避免同时保持
两组全部操作数；检查能否减少spill且不增加LDSM总工作。若需要增加读取、同步或改变
pipeline槽数，必须一起计入审计和配对性能，不能仅凭寄存器数下降判断收益。
尚未实现该后续版本，不计入最佳；不扩大到更多group，也不重启magic-bias。

## 4. 验证与复现

预检、memcheck、synccheck、racecheck各75项有限输入检查，外加每次4项guard缓存修改检查。
覆盖K128/256/384/640/4096、随机/全零/极值/零A scale、指数差13/14/15、code0和奇数组尾部。
code0对FP64参考检查`rtol=1e-5,atol=0`；其他有效输出按`rtol=atol=1e-3`检查。
不安全输入确认实际执行回退；同张量从安全改为不安全或255，确认缓存失效与拒绝。
三项sanitizer均0错误，racecheck同时0警告。sanitizer最大K4096形状为64×128×4096，
完整4096³由24真实样本验证；不把有限sanitizer覆盖说成所有尺寸无错误。

376项相关CPU测试在A100通过；归档另增5项测试复算Event、MSE、guard、SASS与NCU证据。
CUDA及主要脚本commit为`f77b6e71ca83c0f25648b0f6434efc03d8b542b1`，
缓存修改检查commit为`4859b98db4fc67768e43a8418bc2a94d4876e5f5`。
均本地实现并push，再同步A100 fetch/ffmerge；没有重建正式扩展，SHA保持：
`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

```bash
python scripts/probe_roof_pair_scale_shared_codegen.py --output reports/pair_shared_recheck
python scripts/audit_roof_pair_scale_shared_probe.py --directory reports/pair_shared_recheck \
  --best-sass docs/evidence/a100_o378_roof_v51/reports/o378_roof_v51/best_controls.sass \
  > reports/pair_shared_recheck/audit.json
python scripts/validate_roof_pair_scale_shared_probe.py --cubins reports/pair_shared_recheck \
  --output runs/pair_shared_validation
python scripts/benchmark_roof_pair_scale_shared_probe.py --cubins reports/pair_shared_recheck \
  --output runs/pair_shared_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
```

所有输出目录须不存在。原始Event/MSE/环境/SASS/NCU CSV及日志随本文归档；
二进制cubin和`.ncu-rep`保留在A100项目。文本归档SHA：
`ded087023c4c6efda23ecc9ebafd6904c11a06853c305b469f88bfcefefa1236`。

# A100 O3 优化：目标与验收记录

状态：多轮候选开发/验收中，O3 production仍保留原baseline。
目标一未通过前，不开始目标二的5090 magic-bias部署。

## 目标

在同一A100、相同24份4096³真实输入上，让O3相对**当前优化后O1**的配对吞吐比
`T_O1 / T_O3 > 2`，即O3延迟低于O1的一半；不能拿旧O1约4.44ms作为目标分母。
验收不仅看一次最快值，还包括24样本配对、四模式、MSE、同function原生INT4指令审计、
无spill与Compute Sanitizer；保留全部原始计时及CV异常。

## 不变的数学语义

- 原INT8激活无损拆成低U4、高S4，`a=lo+16*hi`，不改变原激活量化。
- 权重仍来自FP16→MXFP4-G128→Q4；本轮不更换定点转换规则或数据。
- 使用U4×S4、S4×S4两个原生INT4 Tensor Core路径，不换成两个INT8 kernel。
- 每G128把两路INT32 partial重构为`low+16*high`，按原group顺序执行FP32 FMA。
- 原输出是唯一逐位回归参考；MSE仍对同一A100的FP32 O0输出计算。

## 首批候选

`csrc/sm80/o3_optimized.cuh`复用O1中有效的优化思路：

1. cp.async与LDSM共用swizzle，明确区分4bit nibble坐标和16-byte搬运坐标。
2. 每CTA/列/G128解码一次W scale，在shared memory中复用；A row scale循环外加载。
3. FP32位解码替代通用ldexp；正常范围guard下使用精确指数位加法。
4. partial留在寄存器；采用精确magic bias，INT8×Q4的G128保守界为
   `abs(partial)<=128*128*8=131072`，仍位于该转换的精确范围。
5. 双缓冲K128/K256 pipeline；K256内部保持两个独立G128 scale，不合并缩放。
6. 编译期CuTe坐标与最终单次输出store。

初筛候选：`64×64×128`、`128×64×128`、`128×64×256`，均256线程。
遇到非normal/极端scale时使用同tile的正常浮点转换路径，显式记录guard状态，不进入软件GEMM。

首轮单样本初筛（不是最终验收）：旧O3约1.858ms，64×64/K128候选约0.613ms，
同轮优化后的O1约1.010ms；O3/O1配对吞吐比约1.64，未达到2倍。三个候选逐位一致、
原生INT4指令及无spill审计通过。128×64/K256用了173register/thread，初筛没有优势。

`swizzle64_ncu`：Duration约606.46us，DRAM8.81%、L1/TEX63.58%、FMA43.0%；
No Eligible43.93%、理论occupancy37.5%。按每issue-active的平均warp stall比率，
barrier约1.914、wait约1.776、math-pipe约1.216、short-scoreboard约0.881，
long-scoreboard约0.295；这些是ratio，不是百分比。

因此继续测试64×32/64×128输出tile，以及CTA完整scale panel预取。后者限制K<=4096，
在**同一个被计时的GEMM kernel内部**加载、解码全部W scales，然后按G128顺序读取；
不跨调用缓存shared memory，也不把这部分处理挪到计时之外。

第二轮单样本64×128/K128约0.595ms，相对同轮O1约1.65倍，仍未达标。
第三轮完整scale panel缓存候选未优于该版本，不能据此切换默认后端。
审计发现缓存初始化中的运行时整数除法生成了`I2F.U32.RP`，而非partial转换退化；
后续将地址分配改为固定8线程/列，消除除法并覆盖小K非4整数倍group的安全加载。
新增`*_exp`候选仅使用指数位scale，保留普通I2F，单独检验magic bias对O3的收益；
O3每G128转换一次，而O1每K32转换一次，两者的转换吞吐压力不能直接类比。
上述均为待验收候选，不改变生产O3默认值。

第四轮单样本：64×128/K128指数位scale＋普通I2F约0.571ms，magic版约0.591ms，
同轮O1约1.034ms；配对吞吐比约1.81，仍未达2倍。384项GPU逐位检查和全部O3
函数指令/无spill审计通过；存在CV超3%的初筛记录，保留原始测量，不视为最终稳定性验收。
缓存地址除法引起的I2F已消除，但缓存候选仍无性能优势。
下一候选保持64×128/K128，采用4×4即16个warp，减少每线程fragment大小；
不得预设更多warp一定更快，继续比较资源使用、正确性和配对结果。

第五轮16-warp候选约0.74ms，慢于8-warp指数位版本约0.554ms；寄存器仍达
81/83个每线程，512线程CTA未获得预期的驻留CTA数优势。240项GPU逐位检查和
指令/无spill审计通过，但性能不满足采纳条件。

新增单partial整数重构候选：G128内先计算high的两个K64 MMA，再将INT32 partial
乘16，最后把low的两个K64 MMA直接累加到同一fragment。两份K64 weight fragment
保留以供两条路径复用；移除第二份完整INT32 partial。所有整数中间结果远小于
INT32界，且最终整数点积与`low+16*high`完全相同；FP32 scale/FMA顺序不变。
它不是改为INT8或FP16 GEMM，仍要求同一正式函数内原生U4×S4及S4×S4 SASS。

第六轮单partial正确性与审计通过，但8-warp版本约0.601ms、16-warp约0.818ms，
慢于同轮普通指数位版本0.562ms。16-warp寄存器降至66，仍高于让两个512-thread CTA
同时驻留的64-register分界；8-warp版本没有降低寄存器数。此处只是资源解释，
并不把降低寄存器数量本身当作性能收益。

下一候选静态展开cp.async搬运次数：根据CTA和16-byte copy大小在编译期分配
每线程复制项，消除threadIdx驱动的运行时循环。保留旧copy loop，同进程比较
8/16-warp、独立/合并partial，不更改数据布局、搬运字节数或G128数学语义。

第七轮静态copy的8-warp独立/合并partial约0.540/0.534ms，相对同轮当前O1
配对吞吐比约1.86/1.85，未达到2倍。500项GPU逐位检查、15项非法输入拒绝
以及原生INT4/无spill审计通过。合并partial静态版125register/thread，仍限制驻留。
下一步仅测试编译期launch-bounds驻留提示：8-warp至少3 CTA、16-warp至少2 CTA；
不改GPU时钟或系统设置，不允许通过寄存器spill换取虚假的occupancy优势。
候选必须重新审计和实测，不预设强制寄存器上限一定有利。

第八轮驻留提示被否决：8-warp/3 CTA出现104-byte stack及LDL/STL，延迟约0.969ms；
16-warp/2 CTA出现24-byte stack及LDL/STL，约0.713ms。虽然300项逐位检查通过，
两者均未通过无spill审计，已移除这两条候选编译入口，负面证据保留在v8目录及Git历史。
资源报告`LOCAL:0`不足以证明没有spill，必须同时检查STACK和SASS load/store。

接下来显式展开双缓冲的两个phase：slot0/slot1作为编译期常量传入相同stage处理，
减少buffer地址计算；仍为两个shared stage、顺序G128 FP32 FMA，奇数stage正确收尾。
不降低精度、不改变K分组、不增加跨CTA归约；继续保持旧路径作为同进程对照。

第九轮400项GPU逐位检查和原生INT4/无spill审计通过，之前spill候选已不再实例化。
phase展开没有带来收益；当前同轮普通静态copy约0.532ms，合并partial约0.551ms，
phase候选约0.544/0.555ms。仍未达到相对当前O1两倍吞吐，不能开始5090阶段。
接下来在24真实样本、同进程交错顺序中比较两种非phase静态copy候选，重新计算MSE
及配对置信区间，作为后续更改tile/fragment布局的可信基线，不将单样本初筛当作达标。

## 24样本配对中间结果（v9，目标仍未完成）

`runs/a100_o3_paired24_v9`：24样本×10轮，warmup5、每轮20次、转换inner100；
每轮交错O0、当前production O1、旧O3和两个候选。原始1200条记录全部保留。
下表延迟是24个样本各自合并轮次后的median再取median；加速比先同样本同轮配对，
因此不必等于两列总体延迟的直接商。CI为24样本配对比率median的bootstrap95%区间。

| 实现 | compute-only median ms | 相对当前O1配对吞吐比 | bootstrap95% CI |
|---|---:|---:|---|
| O0 | 0.610304 | 1.7425 | [1.7352, 1.7861] |
| 当前O1 | 1.063680 | 1.0000 | [1.0000, 1.0000] |
| 旧O3 | 1.911296 | 0.5531 | [0.5414, 0.5613] |
| O3 64×128/K128、指数位scale、普通I2F、静态copy | 0.551936 | **1.8950** | **[1.8373, 1.9217]** |
| 同tile、单partial整数重构、静态copy | 0.575744 | 1.8290 | [1.8024, 1.8511] |

两种新O3在所有24个样本上与旧O3输出逐位一致。O3相对O0的MSE：
median=`0.006653010285119311`，mean=`0.00757884701115429`，新旧完全相同。
当前O1的24样本MSE与`runs/a100_o1_paired_final`中production逐样本完全一致：
median=`9.82338825329489e-9`，mean=`1.1142038139865581e-8`。O1回归指令审计也通过。

稳定性须区分**单轮内**和**多轮合并**：最佳候选240个sample-round的GEMM CV
中位数0.3211%、最大2.3648%，每轮均低于3%；多轮合并统计出现182个stage CV失败项
（包含多个variant及gemm/total两列），反映跨轮时序漂移，不应删掉这些记录后宣称全通过。
不同轮次延迟存在漂移，未锁频；没有同步干扰证据，不能确定归因为其他用户负载。
该结果的上界仍小于2，不能宣称完成目标一。随后完成下述24样本四模式补测，尚未切换默认后端。

### v9 四模式补测

`runs/a100_o3_four_modes_v9`已完成24样本、warmup50、repeats200、inner100、单轮交错。
以下均为该次运行的24样本median延迟，单位ms，不与前一轮paired24的数值混合。
原始384条记录保留；有134个stage CV>=3%项，记录完整不等于稳定性全部通过。

转换阶段（批量摊销）：

| 实现 | W转换 | A转换 | 转换total |
|---|---:|---:|---:|
| O0 | 0.062894 | 0.049034 | 0.112236 |
| 当前O1 | 0.062828 | — | 0.062828 |
| O3静态copy候选 | 0.057600 | 0.041019 | 0.098614 |

Compute-only：

| 实现 | GEMM |
|---|---:|
| O0 | 0.731648 |
| 当前O1 | 1.145344 |
| O3静态copy候选 | 0.642048 |

Cold（total为单次端到端直接测量）：

| 实现 | W转换 | A转换 | GEMM | total |
|---|---:|---:|---:|---:|
| O0 | 0.062904 | 0.049055 | 0.717824 | 0.860160 |
| 当前O1 | 0.062817 | — | 1.152000 | 1.231872 |
| O3静态copy候选 | 0.057615 | 0.040842 | 0.635392 | 0.758784 |

Steady-state（缓存权重转换）：

| 实现 | A转换 | GEMM | total |
|---|---:|---:|---:|
| O0 | 0.048937 | 0.769024 | 0.830976 |
| 当前O1 | — | 1.170944 | 1.177088 |
| O3静态copy候选 | 0.040812 | 0.651264 | 0.705536 |

不同计时轨道及各自median不能简单相加。所有24样本候选输出仍与旧O3逐位一致，
MSE median/mean与上述paired24相同。本轮没有改变W_scale解码的计时归属；
把它移入权重转换是另一个待用户确认的方案，未实施。

下一候选为32×128/K128、2×4共8warp（256threads），保留指数位scale、普通I2F、
静态copy和两份独立partial。目的不是强制寄存器上限，而是从输出tile/warp布局上
减少每线程accumulator；代价是更多CTA及潜在更多W重复加载，必须实测权衡。
候选未通过服务器测试前不作为已完成优化；v9四模式测量结束后才在服务器同步并构建。

第十轮32×128候选通过200项GPU逐位检查（与64×128对照合计）和原生INT4/无spill
审计，寄存器降至80/thread。但延迟约0.556ms，略慢于同轮64×128的0.536ms，
没有满足采用条件。这说明较小fragment的收益仍需抵消CTA数量/重复加载的代价；
不能仅凭occupancy提升认定更快。继续以64×128静态copy作为当前最佳研究基线。

第十一轮准备改进**GEMM内部**完整scale-panel缓存：先将UE8M0字节按
`Swizzle<3,2,5>`存入shared scratch，再以group-major线程顺序解码到FP32/指数位数组。
它避免旧初始化中8个lane向同一个bank写float的地址模式；CPU检查布局双射、
4-byte对齐和固定group下32列bank分布。增加一个初始化barrier及最多4KiB scratch，
是否更快仍需实测；全部初始化保留在GEMM计时内，没有执行待确认的离线预解码方案。

## 当前证据位置（尚非最终验收）

- 原始初筛与逐位验证：`runs/a100_o3_screen_v1`至`runs/a100_o3_screen_v7`。
- PTX、SASS、资源及逐函数审计：`reports/a100_o3_opt/audit_v1`至`audit_v7`。
  v3的缓存初始化整数除法导致严格I2F检查失败，v4已修复；不得把v3写成审计通过。
- NCU完整报告：`reports/a100_o3_opt/swizzle64_ncu.ncu-rep`及
  `exp64x128_ncu.ncu-rep`，同目录保留raw CSV、details和source/SASS CSV。
- 本地已收到上述文件；传输归档`tmp/a100_o3_evidence_v7_clean.tar.gz`的SHA-256为
  `f9705ef0cd552f2893e0a3a884788db2dddcf01e28cd3b1478542d9fadf9d2aa`。
- v7环境文件和audit均记录二进制SHA-256，环境还记录相关CUDA源码SHA-256。
  67项CPU测试通过；v7单样本初筛无CV>=3%记录，但不能替代24样本覆盖。

64×128指数位候选的NCU测得Duration549.15us、REG100、理论occupancy25%、
eligible warp/scheduler0.85、No Eligible52.29%、DRAM9.51%。动态warp指令约154.49M；
wait、math-pipe、barrier每issue-active比率约2.092、0.987、0.769。
这些支持继续研究指令依赖和延迟隐藏，而不是把问题简单归为显存带宽不足。
NCU replay duration不是正式CUDA Event性能统计，不能与另一次O1延迟直接算正式加速比。

## 命令

所有修改先在本地`/root/ADAngel_oyzl`完成并推送GitHub，A100仅同步、构建、运行。
服务器源码目录`/home/zlouyang/ADAngel_oyzl`，写入不得超出`/home/zlouyang`。

```bash
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps
python scripts/benchmark_a100_o3.py --validate --output runs/a100_o3_screen_new
python scripts/audit_a100_o1.py --variant o3 --output reports/a100_o3_audit_new
python -m pytest tests/unit/test_a100_o3_optimization_contract.py -q
```

目标二将单独记录SM120实现、独立构建和5090实测；A100结果不能替代5090验收。

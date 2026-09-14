# A100 O3 优化：目标与验收记录

状态：多轮候选开发/验收中，O3 production仍保留原baseline。
目标一未通过前，不开始目标二的5090 magic-bias部署。

## 目标

在同一A100、相同24份4096³真实输入上，让O3相对**当前优化后O1**的配对吞吐比
`T_O1 / T_O3 > 2`，即O3延迟低于O1的一半；不能拿旧O1约4.44ms作为目标分母。
验收不仅看一次最快值，还包括24样本配对、四模式、MSE、同function原生INT4指令审计、
Compute Sanitizer与资源使用检查；保留全部原始计时及CV异常。

2026-09-14用户确认：若正确性/MSE不受影响，可采用存在少量spill的更快实现。
因此零spill不再单独作为O3淘汰条件；历史严格审计失败记录不回写为通过。
新版审计默认仍严格，可显式传`--variant o3 --allow-spills`把local/stack诊断
列为warnings，同时保留`strict_passed=false`、原始checks、资源及LDL/STL数量。
原生U4/S4指令、禁止INT8退化、cp.async及资源元数据完整性仍是硬性条件。
此开关只改变结构审计政策，不证明数值正确或内存安全；还须通过GPU逐位/MSE、
Compute Sanitizer及24样本同进程性能验收。O1和SM120政策不变。

### 理论目标的解释

A100的非稀疏INT8/INT4峰值分别为624/1248 TOPS，见
[NVIDIA A100数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)。
若O1的整数乘加工作量为C，O3需要低位和高位两路，共2C；仅计算Tensor Core
乘加的理想时间为`2C/P_INT4 = C/P_INT8`，因此不能从INT4峰值直接推出O3快2倍。
两路由同一warp交错发出，独立partial提供指令流水机会，不代表独占两块物理Tensor Core。
单partial候选则主动引入high→乘16→low依赖，换取更少的活跃寄存器。

完整kernel的潜在收益主要来自G128相对G32：K=4096时，逐输出元素的分组
转换/缩放/FP32累加从128次降为32次，同时新增两路整数重构等开销。
“吞吐超过当前O1两倍”是需由配对实测证明的优化目标，不是架构必然保证；
不能通过更换O1分母、移动计时边界或改变FP32分组顺序满足目标。

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

最新`static64x128_ncu`：Duration529.18us，REG128、occupancy25%、eligible0.73、
No Eligible59.27%、动态warp指令126.71M；每issue-active的MIO、short-scoreboard、
wait比率分别约1.391、1.304、1.507。静态copy减少了指令，但并未消除operand等待。
第十一轮缓存候选通过340项GPU逐位检查及同函数原生INT4/无spill审计，但初筛
0.559104ms仍慢于非缓存static的0.533504ms，不采用该候选。
第十轮memcheck和racecheck各完成200项验证，分别为0 errors和0 hazards；
这是验证脚本的小形状/长K覆盖，不等同于所有24个4096³样本都做过sanitizer。

进一步准备atom-stream候选：两个K64的A低/高operand预先进入寄存器，W按
`WN*16`列的小片加载两个K64并复用于低/高MMA，避免整块W双缓冲抬高寄存器数。每次
只保留4个low和4个high的INT32 partial，立刻对对应FP32输出寄存器进行该G128的FMA。
两条INT4 MMA路径、每个输出的group/FMA顺序及量化不变；这是待审计、待实测候选。

第十二轮atom-stream已通过200项GPU逐位检查、同函数原生INT4/无spill审计。
单样本五轮初筛：stream 0.544768ms、static对照0.566272ms、当前O1 1.031168ms；
这不是24样本最终结论，寄存器仍为128，目标尚未证明达成。
下一轮独立测试stream的3 CTA launch-bound及K256搬运候选。若launch-bound导致
spill，必须拒绝该候选，不能为了名义occupancy接受local-memory退化。
K256仍逐个G128缩放/FMA，不把两个group合并缩放。

第十三轮360项逐位检查通过，但3 CTA候选出现REG80/STACK48以及18条LDL、18条STL，
审计失败，已拒绝并移除该候选。与此同时，显式minBlocks=1并不等价于省略该参数：
static和stream的REG从128变为135，单SM驻留由2 CTA降为1 CTA，初筛约0.724/0.725ms。
该轮不能用于声称原static回退或stream加速。K256候选REG194、无spill、约0.616ms，
也受此配置影响；下一轮恢复原单参数launch_bounds，重新对照K256，不混用两轮延迟。
原始失败审计及结果保留在`audit_v13`和`runs/a100_o3_screen_v13`。

第十四轮恢复后260项GPU逐位检查和完整O3审计通过。static/stream K128/K256分别
约0.559104/0.570368/0.593920ms，当前O1约1.051648ms。K128恢复REG128，K256为
REG130、无spill，仍越过2 CTA的寄存器上限。准备单独的`swizzled_bound2`入口，
仅对K256指定minBlocks=2；所有其他入口保持原单参数launch_bounds。
两个入口复用同一个forceinline函数体，量化及G128/FMA语义不变；需重新审计和实测。

第十五轮独立bound2入口初筛0.478208ms，当前O1约1.071104ms，单样本配对比2.1761。
320项逐位检查通过，但快速路径REG128/STACK8（fallback STACK16），各有2 LDL/2 STL，
因此**尚未通过验收**。SASS中spill对应一项FP32输出accumulator，而不是量化误差。
下一轮仅在bound2入口禁止两个G128的编译期展开，保留K256搬运和顺序G128循环，
尝试缩短operand/地址临时量生命周期；不能用这次有spill的初筛宣称目标完成。

第十六轮串行G128版本通过220项GPU逐位检查，但仍为REG128/STACK8、2 LDL/2 STL，
审计未通过，初筛0.502784ms。下一轮恢复G128展开，仅在bound2入口将W寄存器片
从32列缩为16列（每warp一个N8 atom），用LDSM x2代替x4，减少同时存活的W寄存器。
该变化增加小片加载指令数，是否划算仍由实测决定；保持两条INT4路径和FP32 FMA顺序。

第十七轮160项逐位检查通过，初筛0.487424ms，但快速路径仍有STL.64/LDL.64及STACK8。
SASS显示这次溢出的是预取激活所用的64位地址，不再是partial/output。
下一轮仅在bound2入口使用受容量检查保护的u32 packed-byte偏移，完整计算偏移后再
加上64位基地址。两份packed A合计和packed W均须小于2^32字节，否则明确报错；
4096³满足该条件。其他候选继续使用原指针表达式。

第十八轮160项逐位检查通过，初筛0.491008ms；快速路径STACK32和7 LDL/7 STL，
比原地址方式更差，故撤回compact offsets及其容量限制。
下一轮回到32列W片/LDSM x4，在每个W片消费完成后增加warp-scoped同步，限制编译器
跨片提前加载造成的寄存器存活范围扩张。所有32个lane执行相同固定片数，
没有partial shared-memory中转，也没有新增CTA级barrier；实际收益及无spill仍需验证。

第十九轮160项逐位检查通过，初筛0.488448ms，但快速路径STACK16/2 LDL/2 STL，
fallback STACK8，仍未通过无spill门槛。stream-bound2候选从当前入口中移除，历史
提交和全部原始结果保留。下一轮测试K256的`exp_merge_static_bound2`：在每个G128内
按high→乘16→low的整数重构方式复用一个完整INT32 fragment（该恒等式已穷举/随机验证），
保持原FP32 group/FMA顺序、原指针地址和单独2 CTA入口，不增加warp小片同步。

第二十轮260项GPU逐位检查通过，单样本MSE仍为0.0005745973478203796。
完整fragment合并候选约0.570368ms，快速路径STACK32/8 LDL/8 STL、fallback
STACK48/16 LDL/16 STL，均不满足无spill要求，也未显示性能优势，故撤回该入口。
下一候选`exp_merge_static_stream_bound2`在32列W片内逐MMA atom重构：
仅四个INT32寄存器先积累两个high K64，乘16后加入两个low K64，再做原FP32 FMA。
保留双K64 A fragment、W片复用、warp片边界和K256内两个独立G128；不更改量化、
不移动转换计时。该候选须重新验证，不能以第二十轮结果代表它。

第二十一轮单MMA atom合并：160项GPU逐位检查通过，单样本配对初筛2.0186倍；
快速路径REG128/STACK8/2 LDL/2 STL，fallback STACK16。按用户确认后的政策，
原生INT4/cp.async审计通过并保留spill警告，`strict_passed=false`。
24样本、每样本3轮、warmup50/repeats200的正式配对结果为1.91953倍，bootstrap
95%区间[1.91252,1.92881]，未达2倍；O3 GEMM样本中位数0.595968ms，当前O1
1.149952ms。二者中位数之比并非配对加速比的定义，不可混用。
全部24样本新旧O3输出逐位一致，MSE median=0.006653010285119311、
mean=0.00757884701115429；保留74条跨轮汇总阶段CV异常，不删样本。
O1独立指令审计通过，74项单元测试通过；memcheck/racecheck各完成60项边界/形状
验证，分别为0 errors/0 hazards。另对完整4096³各执行一次memcheck及racecheck，
同样为0 errors/0 hazards（均error-exitcode=99，实际退出码0）。这些检查证明当前
覆盖范围内没有检测到内存错误或竞争，不代表已达到性能目标；四种性能口径仍待补齐。
原始证据保存在`runs/a100_o3_paired24_v21`和`reports/a100_o3_opt/*v21*`，传输归档
`tmp/a100_o3_evidence_v21.tar.gz`的SHA-256为
`d3fbf9b519b670183266f127820356912bb596058ee0ca431b10091f6e273339`。

既然用户允许少量spill，恢复第十五轮的独立partial/K256/2CTA候选，保留
32列W片、LDSM x4、原始指针运算且不增加warp片边界，与合并候选同进程比较。
候选名为`o3_swizzle_64x128_k256_exp_static_stream_bound2`；准确版本以git commit、
源码和binary SHA为准（中间历史曾对同名入口试验warp边界）。不改production默认值。

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

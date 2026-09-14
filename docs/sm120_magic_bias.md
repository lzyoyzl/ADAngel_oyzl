# RTX 5090：O1/O3 exact magic-bias 实验

状态：24样本正式配对完成，两项magic候选均未加速；81项CPU测试、216项GPU逐位检查、
129项集成测试和专门指令审计通过。memcheck零错误，但racecheck报告旧/新O1及O3共享scale
竞争，v4安全验收不能标为通过。用户已同意修复：本地已补充post-store release，
等待5090重新编译、racecheck及修复前后逐位回归；当前production实现选择未切换。
目标一A100的结果见`a100_o3_optimization.md`；不得将其速度收益直接套用到5090。
用户确认本轮同时测试O1和O3。

## 唯一改变的计算环节

```cpp
float(partial)
// 候选替换为：
__fadd_rn(__int_as_float(0x4b400000 + partial), -12582912.0f)
```

O1每K32的partial满足`abs(partial)<=32*128*12=49152`；O3每G128在
`low+16*high`重构后满足`abs(partial)<=128*128*8=131072`。
两者都在bias附近FP32 ULP=1的区间内，转换结果逐位等于普通INT32→FP32转换，
包含负整数和零。CPU穷举O3整个保守范围，同时覆盖O1范围。

不修改权重量化、激活拆分、行/列scale位置、FMA次序、CTA、TMA或warp分工，
也不同时移植A100的指数位scale优化，以隔离本次转换的收益。
共享helper中的编译期bound不是运行时范围检测；调用点的范围由量化值域和group长度保证。

| 后端 | 旧实现 | 候选 | CTA |
|---|---|---|---|
| O1 | `register_128x64_k64_scale_shared_row_dedup` | 旧名加`_magic` | 128×64×64 |
| O3 | `m64_n32_k128_aligned_factor_16w` | 旧名加`_magic` | 64×32×128 |

O3继续保持当前SM120 legacy U4/S4 PTX路径，其CUDA12.8 SASS为INT8 IMMA加位操作，
本实验不会把它变成原生INT4硬件指令。旧实现继续可调用，不将A100版本覆盖成5090版本。

## 已确认的编译与正确性证据

证据位于`reports/sm120_magic/audit_v4`、`unit_v4.log`和
`runs/sm120_magic_smoke_v2/validation.json`。CUDA候选编译自`dfb02b3`；
`36c2d2d`只修订审计、测试和说明，不需要再次编译CUDA。

| 项目（静态SASS计数） | O1旧 | O1 magic | O3旧 | O3 magic |
|---|---:|---:|---:|---:|
| I2FP | 36 | 4 | 198 | 66 |
| I2F | 2 | 2 | 33 | 33 |
| FADD | 0 | 32 | 0 | 132 |
| IADD3 | 58 | 90 | 144 | 276 |
| FFMA | 32 | 32 | 132 | 132 |
| IMMA | 8 | 8 | 4 | 4 |
| UTMALDG | 4 | 4 | 99 | 99 |
| 寄存器/线程 | 56 | 56 | 56 | 56 |
| STACK / LOCAL | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

这些是静态指令位置计数，不能当成动态执行次数。新旧变化恰为partial I2FP
替换成IADD3+FADD；producer解码scale的转换保留。所有4个内核均没有LDL/STL。
旧O1/O3内核机器指令与编译前备份分别逐字一致，未通过改变旧基线制造加速。
GPU验证覆盖不同M/N/K、全部4种模式、饱和和编码模式，新旧输出FP32逐位一致。
单样本冒烟不足以判断性能收益，正式结论须等待24样本配对、四模式及内存安全结果。

## 24样本专门Compute-only配对（v4）

`runs/sm120_magic_paired_v4`：24样本×5实现×3轮，共360条唯一记录；warmup50、
repeats200、inner100，所有原始计时完整，保留2个跨轮汇总CV>=3%的stage条目。
速度比定义为同样本/同轮`T_old/T_magic`，再统计配对中位数，不用不同run相除。

| 实现 | GEMM median ms | 相对各自旧实现的配对速度比 | Bootstrap 95% CI |
|---|---:|---:|---|
| O0参考 | 0.765672 | — | — |
| 旧O1 | 0.625736 | 1 | — |
| O1 magic | 0.657000 | 0.957345 | [0.951510, 0.966610] |
| 旧O3 | 2.052576 | 1 | — |
| O3 magic | 2.061840 | 0.994982 | [0.994897, 0.995099] |

两项CI均低于1，不能以减少I2FP为由宣称加速，当前不晋升任一magic候选。
所有新旧同后端输出仍逐位相同，其对O0的MSE也完全相同：

| 同后端新旧两实现 | MSE median | MSE mean |
|---|---:|---:|
| O1 | 9.823381579891711e-9 | 1.1141469767717465e-8 |
| O3 | 0.006653010195220884 | 0.007578844400053298 |

### 安全检查的未解决项

`memcheck_v4.log`为0 errors；`racecheck_v4.log`最终为100条显示hazards、
250 errors、0 warnings。已显示的报告同时涉及旧O1和magic O1的
`column_scale_factor`写入（`o1_gemm.cu:895`）与读取（`:1017`）。
这不能被逐位相同或MSE通过抵消，也不能未经进一步核实称为工具误报。
旧O1机器指令未改变，表明风险不只出现在magic新增转换上；是否单独修复原有同步
已向用户确认。在取得结论前保留原始日志、默认不晋升。
单独O3 racecheck完成96项数值检查后仍报1050 errors，显示100条hazards，定位到
`o3_gemm.cu:559`写入与`:983`读取。因此不能认为第一次混合报告仅出现O1就代表O3无风险。
已补充询问是否单独修复O1/O3的同步并重新测试；暂不把同步修改混入magic转换实验。

### v4同步风险的源码与内存模型佐证

修复前O1和O3的producer实际顺序为：

```text
producer_acquire：等候空stage，并对full barrier执行arrive_and_expect_tx
→ producer warp写入普通shared-memory scale
→ __threadfence_block + __syncwarp
→ 发出A/W的TMA copy
consumer：等候full barrier完成 → 读取A/W以及shared scale
```

关键不在于缺少TMA等待，而在于scale的发布发生在full barrier的release arrival之后。
固定CUTLASS源码`include/cutlass/pipeline/sm90_pipeline.hpp`中的
`PipelineTmaAsync::producer_acquire(stage, phase)`先等empty barrier，随后由leader
调用`full_barrier_ptr_[stage].arrive_and_expect_tx(...)`。O1/O3均在该调用返回后才写scale。

CUDA12.8 PTX8.7规定，bulk异步拷贝的隐式complete-tx只为该异步操作自身的访问建立顺序，
不会传递发布发起线程之前的其他访问；mbarrier等待也不为release arrival之后的普通访问
提供该发布保证。因此TMA数据完成，不能直接推导普通shared scale也已通过同一barrier
正确发布给consumer。[官方PTX内存模型及mbarrier说明](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html)

这里的`__syncwarp`只同步producer warp；现有fence与warp同步没有补上consumer所需的
跨warp发布/获取关系。这与racecheck定位的scale读写hazard一致，不能仅以数值回归通过
判定为误报。它不证明已保存的所有输出都是错误的，但意味着不能保证所有调度下正确。

拟议最小修复是：保留empty-stage保护，在所有scale写入结束后增加明确的release发布，
并让consumer必须等到该发布及TMA完成后才能使用stage。可评估额外full-barrier arrival
或独立的每stage scale-ready barrier；必须同时核对arrival计数、phase及stage复用，
不能简单把scale写入移到`producer_acquire`前，否则可能覆盖尚被consumer使用的stage。
以上为修复方案约束；下面记录用户确认后的具体实现。

旧实现与magic候选应用同一个同步修复，然后重新进行racecheck、MSE逐位回归、
24样本配对和四模式计时；保留v4原始证据，不能将同步修复的收益归因于magic-bias。

### 同步修复v7（已实现，等待GPU验收）

v5尝试让lane0在写后额外arrival：编译/ISA审计通过，但racecheck仍报告其他writer的
scale竞争，因此不验收v5，也不删除失败日志。v6改为每个producer lane直接发布自己的写入，
仍未通过。进一步启用`--racecheck-report hazard`定位到WAR（读后写）：例如consumer线程257
读过的shared地址0x9404被producer线程1覆盖，缺少stage复用方向的直接acquire。
v7同时让全部32个producer线程执行`producer_acquire`，只有lane0为leader登记事务，
从而每个writer在覆盖stage前都直接获取empty barrier、写入后都直接release full barrier。
O1的K32 shared-scale与K64 shared-scale路径，以及O3共同TMA路径，将full barrier
每stage的预期arrival数从1改成33：一次在`producer_acquire`登记TMA事务；
其余32次由producer warp各lane在自身scale/correction写入后执行
`cutlass::arch::ClusterBarrier::arrive`（默认release）。既有consumer wait是获取端。
这里的33是arrival计数，不增加producer warp；没有改动CUTLASS源码。

full barrier只有在全部33次arrival和TMA事务均完成后才放行；empty barrier仍保护stage复用。
不使用shared scale的O1保留一次arrival。数学公式、CTA、stage数、量化和MMA选择均不变。
元数据新增`scale_publication=per_writer_acquire_release_v3`和
`full_barrier_arrivals_per_stage=33`，区分同名kernel修复前后的运行。

`benchmark_sm120_magic.py --snapshot-only`可保存24样本×5实现的输出SHA-256与MSE；
修复后传入`--compare-snapshot <修复前目录>/snapshot.json`，要求全部逐位指纹和MSE一致。
指纹比较不能替代racecheck或独立语义参考，两者均需通过。

## 针对性NCU观察（v4，非正式计时）

四份报告为`reports/sm120_magic/ncu_{o1,o1_magic,o3,o3_magic}_v4.ncu-rep`，
附details与raw CSV。采集SpeedOfLight、SchedulerStats、WarpStateStats、InstructionStats，
不锁频，每次仅选定一项旧/新内核，warmup50后捕获一次；不是`--set full`。

| 实现 | NCU duration | DRAM throughput | Eligible warps/scheduler | Issued warps/scheduler | Executed instructions |
|---|---:|---:|---:|---:|---:|
| O1 | 622.27 us | 6.18% | 1.36 | 0.42 | 454,310,806 |
| O1 magic | 642.91 us | 6.01% | 1.81 | 0.49 | 521,038,968 |
| O3 | 约1.94 ms | 1.73% | 3.59 | 0.67 | 2,507,198,339 |
| O3 magic | 约1.95 ms | 1.73% | 3.60 | 0.68 | 2,524,187,276 |

O1动态指令增加约14.7%，O3增加约0.68%。magic把一次转换变成整数加法和浮点加法，
不是删除所有代价：O1虽然eligible与issue改善，工作量也变多，最终延迟没有改善；
O3的其他指令开销占比较大，此改动影响较小。该证据支持“管线压力转移不一定带来收益”，
不能仅凭低DRAM吞吐就精确判定某条执行管线饱和。A100上的正收益不能直接迁移为5090结论。
这些NCU指标是报告原名，不能混同为线程指令或Tensor Core数学吞吐；
重放duration也不能与另一轮CUDA Event延迟交叉计算正式速度比。
上述分析暂基于未修复同步的实现，不能作为安全认证；若修复同步，需新旧同条件重新测量。

本轮原始运行、所有成功/失败审计、sanitizer日志与四份NCU报告打包为
`tmp/sm120_magic_evidence_v4.tar.gz`，SHA-256：
`9dea67f06d535b6cf93b8a9d612aa2b78fc944ff6aba9f53acc15bc3c76164a1`。
四模式计时暂未执行完备；不能把本节compute-only数据改名为cold或steady结果。

## 验收与计时

- 合成输入覆盖全零、随机、饱和、正负交替、E2M1全部编码及零scale；新旧输出FP32逐位一致。
- 24真实样本重新计算对O0的FP64 reduction MSE；新旧同后端MSE必须完全相同。
- 同进程、相同输入、交错顺序、warmup50/repeats200/3轮；保留所有CV及离群值，不锁频。
- conversion-only独立批量摊销inner100；compute-only GEMM、cold及steady total均保留直接计时。
- 同一候选function必须有TMA、正确MMA语义，partial的I2FP由FADD/IADD替换；
  producer用于scale解码的I2F/I2FP保留，不能声称整个kernel没有I2F。审计保留全局计数，
  同时核对新旧同function的I2FP减少数与FADD/IADD3增加数相等，以及MMA/TMA/FFMA不变；
  结合只改partial转换的源码和逐位GPU回归建立证据，资源和spill检查保留。
  不用“第一个MMA之后”的线性PC区间推断执行顺序，ptxas可重排带回跳的基本块。
- 逐后端比较旧/新配对速度。正确性和安全是硬条件；无明确收益则保持旧production。
- memcheck/racecheck与现有O0—O4集成测试分别执行，不与正式GPU计时并行。

## 服务器命令

源码先在本地修改并推送GitHub，再由服务器fetch/merge。5090所有新增文件、构建输出与缓存
限定在`/home/zlouyang/oyzl/ADAngel_oyzl`；不安装或修改该目录以外的环境。
以下使用已经存在的`adangel-sm120` Python与CUDA12.8，不重新配置环境：

```bash
cd /home/zlouyang/oyzl/ADAngel_oyzl
export PYTHONDONTWRITEBYTECODE=1
mkdir -p tmp/sm120_magic .cache/magic reports/sm120_magic
export TMPDIR="$PWD/tmp/sm120_magic"
export XDG_CACHE_HOME="$PWD/.cache/magic"
export CUDA_CACHE_PATH="$PWD/.cache/magic/cuda"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/magic/torch_extensions"
export PIP_CACHE_DIR="$PWD/.cache/magic/pip"
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm120 MAX_JOBS=4 \
  python setup.py build_ext --inplace

python scripts/benchmark_sm120_magic.py --validate-only --output runs/sm120_magic_validation
python scripts/audit_sm120_magic.py --output reports/sm120_magic/audit
python scripts/benchmark_sm120_magic.py --validate --samples 0 \
  --warmup 50 --repeats 200 --rounds 3 --inner 100 --output runs/sm120_magic_paired
python scripts/summarize_a100_o1.py --input runs/sm120_magic_paired \
  --output runs/sm120_magic_paired/aggregate.json
python scripts/benchmark_sm120_magic.py --all-modes --samples 0 \
  --warmup 50 --repeats 200 --rounds 1 --inner 100 --output runs/sm120_magic_four_modes
```

总结脚本名称沿用A100历史命名，但只读取同一run的原始记录、不调用GPU；新增O3配对参考
使SM120候选分别和各自旧后端比较。生产结果必须记录真实implementation，不能混用候选和旧run。

# A100 O1 优化与验收

当前选定实现：SM80 `production` 在4096³使用 `swizzle_128x64_k128_magic`；旧实现
保留为显式 `baseline`。本轮目标“快于O0”尚未达成，不将明显快于旧O1误写成快于O0。

目标是同一 A100、24 个相同输入的 O1 GEMM-only 延迟低于 cuBLASLt FP16 O0；
不保证通过削弱数值要求或选择性删除计时值达到目标。原 SM120/5090 后端保持独立。

## 不改变的语义

- E2M1 精确映射为 `2*E2M1` INT8 基值；A 仍为原始 prepared INT8。
- 每 K32 独立 INT32 partial，按原顺序执行
  `acc = fma(float(partial), A_scale * (decode(W_scale)*0.5), acc)`。
- partial 和最终 FP32 accumulator 留在寄存器；每个输出元素只写一次。
- 公共预处理不计时；独立转换批量摊销，cold/steady total 单次直接计时。
- 保留 `baseline` 作为同进程配对对照，不重采或重新量化 trace。

## 当前实现

`csrc/sm80/o1_optimized.cuh` 独立实现 SM80 cp.async 双缓冲，4096³固定配置为
CTA `128×64×128`、256线程（8warp）、2个pipeline stage。K128是搬运粒度，
不是scale粒度；每个stage内依次计算4个K32 partial。

这不是将5090的TMA直接移植到A100；A100版本使用其硬件支持的
[global→shared异步copy](https://docs.nvidia.com/cuda/archive/12.8.0/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory)，
warp协同搬运与计算。5090仍使用原来的独立SM120后端。

优化包括：

1. global→shared 的 16-byte copy 与 shared→register LDSM 共用 CuTe XOR swizzle 布局；
2. W scale 每 CTA/column/K32 仅解码一次，在 stage-local shared buffer 中向输出行复用；
3. row A scale 在 K 主循环外加载；不将其乘法移至最终输出，以维持原 FMA 递推；
4. K128 stage 内处理4个K32，不合并不同scale的partial；
5. CuTe fragment坐标保持为编译期常量，避免生成不必要的动态地址计算；
6. 精确bias-bit转换移除常规I2F，具体证明见下节；没有运行时autotune。

消除尾部冗余CTA barrier：
下一轮开头 wait+barrier 已保护旧槽位全部 reader，下一次 prefetch 在该 barrier 后才覆盖。
内存/同步正确性还必须通过Compute Sanitizer，而不是仅依赖源码推理。

UE8M0/2 通过 FP32 位模式精确生成，包括 code 0/1 的 subnormal；不再调用通用 ldexpf。
scale 以每列连续2/4字节读取，shared store 按列合并，避免转置式写入的 bank 冲突。

当前实现将 `A_scale * 2^(code-128)` 改为指数位加法。计时前严格检查所有
A_scale 是正 normal，且全部可能乘积也为有限 normal；否则明确记录
`exponent_scale_fast_path=false` 并执行相同 tile 的正常 FP32 FMUL kernel。
正常范围内该操作逐位等价，不改变最后 FMA 或 group 顺序，不是把 A_scale 移到末尾。
该操作把部分scale工作从FP32乘法换为整数加法，但没有改变FMA递推。

### 精确 partial 转换（magic bias）

O1 每个 K32 的整数点积满足 `abs(partial) <= 32*128*12 = 49152`。在 FP32 数值
`12582912 = 1.5*2^23` 附近，ULP 恰为1。因此

```cpp
float value = __fadd_rn(__int_as_float(0x4b400000 + partial), -12582912.0f);
```

对该完整范围逐位等价于 `float(partial)`，包括负整数与零。它不是近似量化，也不改变
group/FMA 顺序。其目的在于把常规 I2F 转换从 XU 管线换成 IADD+FADD。
CPU 单元测试穷举98305个可能整数；GPU 再做新旧输出逐位对照、MSE回归和SASS审计。
`_magic` 候选还使用上述安全指数位 scale；不满足 guard 时明确回到普通精确 kernel。

### NCU 定位证据

不锁频 `baseline_full_ncu` 的旧 O1 Duration 约4.36ms，动态 warp 指令为1,429,766,144，
shared load bank conflicts 为150,994,944。首个 swizzle/shared-scale 候选 Duration约1.19ms，
动态指令427,556,864，DRAM throughput约7.56%，XU执行管线的 elapsed峰值占比约77.31%。
这里 NCU 指标用于定位原因，性能验收仍用普通 CUDA Event，不将 profiler Duration混入主表。

最终 `production_final_ncu`（`--set full --clock-control none`）结果：

| 指标 | 旧 O1 | 当前 O1 |
|---|---:|---:|
| NCU Duration | 约4.36 ms | 0.985760 ms |
| 动态warp指令数 | 1,429,766,144 | 378,376,192 |
| shared load bank conflicts | 150,994,944 | 198,184 |
| DRAM throughput | 1.99% | 10.08% |
| 当前最高利用率执行pipeline | ALU（约38.2%） | FMA（74.4%） |

当前XU指令吞吐占比为0；SASS也直接验证I2F为0，不只根据NCU百分比推测。
FMA pipeline统计还包括部分IMAD等整数指令，不能把74.4%解读成“74.4%的Tensor Core算力”。

4096³的 K32 partial 转换数为 `4096*4096*128 = 2,147,483,648`，不是 Tensor Core MMA
指令数。常规类型转换的吞吐应与INT8 Tensor Core峰值分开分析；参见
[CUDA 12.8 Arithmetic Instructions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#arithmetic-instructions)。

动态shared-memory opt-in及scale guard在计时前完成。4096³实际kernel使用128个
register/thread、51200字节动态shared memory；未发现local/stack/spill。
更小的合法输入由明确的固定规则选择64×64/K128或64×64/K64；显式候选不对齐则报错。
元数据同时记录 `requested_implementation`、实际 `implementation`、tile及guard状态。
O3的`production`仍解析为其原生INT4基线，未改动O3。

## 本地开发、同步、编译

所有源码修改在本地 `/root/ADAngel_oyzl` 完成并推送 GitHub。服务器仅 fetch/快进合并
后构建。若服务器 GitHub TLS 失败，可传输已推送提交的 Git bundle，再 verify/fetch/
快进合并。不得直接在服务器修改实现。A100 所有写入限制在 `/home/zlouyang`。

```bash
cd /home/zlouyang/ADAngel_oyzl
export TMPDIR=/home/zlouyang/tmp
export PIP_CACHE_DIR=/home/zlouyang/.cache/pip
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/home/zlouyang/miniconda3/envs/adangel-a100/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps
```

## 验证与测量

```bash
python scripts/benchmark_a100_o1.py --validate-only --impl production \
  --output runs/a100_o1_correctness_new
python scripts/audit_a100_o1.py --output reports/a100_o1_audit_new
python -m pytest tests/unit/test_a100_o1_optimization_contract.py -q

# 同进程24样本配对：每sample/variant累计10×20=200次，交错测量。
python scripts/benchmark_a100_o1.py --samples 0 --rounds 10 --warmup 5 --repeats 20 \
  --impl baseline production --output runs/a100_o1_paired_new
python scripts/benchmark_a100_o1.py --samples 0 --rounds 1 --warmup 50 --repeats 200 \
  --all-modes --impl production --output runs/a100_o1_four_modes_new
python scripts/summarize_a100_o1.py --input runs/a100_o1_four_modes_new \
  --output runs/a100_o1_four_modes_new/aggregate.json
```

脚本保存逐轮原始 timing、kernel 元数据、二进制 SHA、输入 manifest、运行环境与 GPU
状态快照。候选输出必须与旧 O1 逐元素完全一致，并重新计算 FP64 reduction 的 MSE vs O0。
所有 CV 异常保留，汇报 median/mean/P5/P95/CV，不把失败轮删除后标记通过。

审计必须在同一个实际候选 function 内确认 PTX cp.async+S8 MMA、SASS LDGSTS+IMMA，
并检查 local/stack/spill。profiling 不替代普通 CUDA Event 性能测量。

## 结果

### 环境与统计口径

- NVIDIA A100 PCIe 40GB；CUDA12.8.93、PyTorch2.7.1+cu128、driver570.124.06。
- 核心代码提交 `bfc8959050dcc9219ddfae4c17d607c570517bad`；后续仅补充测试和文档。
- 24个原有Llama-2-7B样本，M=N=K=4096，不重新量化或采集；FP32输出。
- 下表Median/Mean均在“每个样本的测量median”上跨24样本计算，不是把所有kernel
  的单次延迟直接混合求均值。逐次原始延迟和单样本mean也在JSONL中保留。
- MSE单独按每个4096×4096输出计算，使用FP64 reduction，然后汇总24个MSE。

### 同进程配对GEMM对照

`runs/a100_o1_paired_final`：每sample/variant交错10轮，每轮预热5次、测量20次，
总计200次。全部720条round记录保留，不丢弃异常轮。

| 实现 | Median ms | Mean ms | 对旧O1配对加速比 |
|---|---:|---:|---:|
| O0 cuBLASLt FP16 | 0.606720 | 0.602048 | — |
| 旧O1 baseline | 4.444928 | 4.450197 | 1.00× |
| 当前O1 production | 0.993280 | 0.995733 | 4.4723× |

当前/旧O1的配对加速比bootstrap95%区间为 `[4.4441,4.4894]`；当前O1相对O0的
配对吞吐比为 `0.6093×`，区间 `[0.6062,0.6117]`。因此**显著快于旧O1，但没有超过O0**。
bootstrap以样本为单位，不能替代跨独立运行的环境稳定性验证。

本次跨10轮合并后，有72个“sample×implementation×stage”的CV≥3%条目：旧O1有24个，
新O1有48个（gemm/total分别计数）；max CV分别3.61%和8.37%。O0 max CV约1.67%。
不能将此配对运行描述成所有阶段均通过严格CV阈值。

### 四种计时模式

`runs/a100_o1_four_modes_final`：每sample/variant/mode预热50次、测量200次，
转换阶段inner=100。24样本×O0/当前O1×4模式，共192条记录。
这是另一轮完整运行，不与上面的配对运行混合计算比值。

**1. Conversion-only：独立转换批量摊销**

| 操作 | Median ms | Mean ms |
|---|---:|---:|
| O0 W：MXFP4→FP16 | 0.062976 | 0.062897 |
| O0 A：INT8→FP16 | 0.049224 | 0.049151 |
| O0转换total | 0.112666 | 0.112429 |
| O1 W：MXFP4→INT8基值 | 0.063601 | 0.063497 |
| O1转换total | 0.063601 | 0.063497 |

O1无需在线激活转换。该表不包含公共FP16→INT8/MXFP4预处理。

**2. Compute-only / GEMM-only：输入提前转换**

| 实现 | GEMM Median ms | GEMM Mean ms |
|---|---:|---:|
| O0 | 0.747520 | 0.744085 |
| 当前O1 | 1.158144 | 1.158037 |

**3. Cold：当前配置全部转换+GEMM，单次直接计时**

| 实现 | Cold total Median ms | Cold total Mean ms |
|---|---:|---:|
| O0 | 0.854784 | 0.847445 |
| 当前O1 | 1.226240 | 1.228437 |

**4. Steady-state：权重转换缓存，单次直接计时**

| 实现 | Steady total Median ms | Steady total Mean ms |
|---|---:|---:|
| O0（A反量化+GEMM） | 0.782336 | 0.778731 |
| 当前O1（无需A转换，仅GEMM路径） | 1.164800 | 1.165653 |

阶段转换使用批量摊销，而total直接记录单次区间，并包含相应事件/执行间隙。
各模式的缓存/运行条件不同，median也不具可加性，因此不能把conversion-only与
compute-only的median相加来替代cold或steady total。

四模式运行中，**当前O1的所有单样本阶段CV均<3%**；O0有86个CV≥3%的阶段条目：
cold31、compute-only24、steady-state31，最高约6.45%。GPU快照记录到其他计算进程，
温度约51→70°C、SM频率在部分时刻由1410降至1380MHz。这些是可观察的运行条件变化，
不是对每一个离群值的唯一因果证明。没有等待GPU空闲、锁频、停止他人任务或过滤数据。
四模式绝对延迟比配对运行更高，应保留这一差异，不能只选较快的一轮作为唯一结果。

### MSE及正确性

| 实现 | MSE vs O0 Median | MSE vs O0 Mean |
|---|---:|---:|
| O0 | 0 | 0 |
| 旧O1 | 9.8233882533e-9 | 1.1142038140e-8 |
| 当前O1 | 9.8233882533e-9 | 1.1142038140e-8 |

当前O1与旧O1在24个样本中逐位一致（通过int32视图比较输出位模式），不是仅满足
`rtol/atol`。四模式运行的MSE也相同。此处参考是A100自己的O0，不混用5090的输出。

- 60项单元测试通过；bias转换完整98305个整数范围穷举通过。
- 包括小尺寸/K64分支在内的128项生产路径逐位对照通过；非法scale输入按预期拒绝。
- 96项O1/O3语义回归通过，O3保留原实现。
- Compute Sanitizer memcheck与racecheck均为0错误/0hazard。
- 17个实例化优化function分别审计通过；同一function中有cp.async/LDGSTS和INT8 IMMA。
- 当前4096³生产function：128register/thread，LOCAL=0、STACK=0、LDL/STL=0；
  静态SASS计数I2F=0、FADD=128、FFMA=128、IMMA=32。这是静态函数计数，不是动态执行次数。
- 本轮未改动`csrc/sm120/`、5090 bindings及默认sm120构建目标；没有在5090上部署本轮代码。

### 可复核的证据

仓库内的[证据目录](evidence/a100_o1_optimization/)包含配对/四模式的原始JSONL、
aggregate、environment，及正确性、sanitizer、审计和NCU文本摘要。
完整NCU二进制、PTX/SASS及开发日志保存在本地与A100的
`reports/a100_o1_opt/`；最终运行数据在`runs/a100_o1_*final/`。
GitHub只提交必要的文本证据，不提交大二进制NCU报告或其他用户的进程路径。

## 为什么目前仍没有超过O0

1. O0在GEMM之前已把A/W转换成FP16，主kernel可以连续沿整个K轴执行FP16 Tensor
   Core累加。O1的INT8 MMA不接收K32 scale，因此每32个元素之后仍须进行软件缩放。
2. 当前O1在4096³中仍有21.47亿个“输出元素×K32 group”的partial转换与FP32累加。
   bias-bit技巧消除了I2F，但不是消除了这些工作，而是用整数加法和FADD代替I2F。
   每个partial还需要scale构造和FP32 FMA；不能只按INT8 Tensor Core峰值估计总耗时。
3. 最终NCU中XU占比降到0，瓶颈转向FMA/指令发射及寄存器约束：FMA最高
   利用率约74.4%，每scheduler活跃warp约3.88，约35.01%的周期没有eligible warp。
   DRAM throughput仅约10.08%，不支持“主要因为显存带宽不足”的结论。
4. K128 pipeline减少搬运/同步次数，但没有取消其中4个K32的独立scale；本轮没有
   用跨group合并或改变求和顺序掩盖这部分代价。寄存器数也限制了可驻留CTA数量。

这些证据解释当前实现，**不是证明所有合法O1实现都不可能快于O0**。在保留逐位一致
要求的情况下，进一步工作应聚焦寄存器/指令调度与shared scale读复用，而不是继续
调高转换批量计时次数，或改用较慢的O0来人为达到目标。

若后续允许改变FP32舍入路径，可单独研究把A_scale放在最终输出处、或利用相同scale
进行数学等价的分组重构；这需要新的MSE验收，且必须明确不能保证与旧O1逐位一致。
本轮没有实施这类重结合优化，也没有放松误差阈值。

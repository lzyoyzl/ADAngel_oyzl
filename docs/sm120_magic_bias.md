# RTX 5090：O1/O3 exact magic-bias 实验

状态：候选已实现，等待SM120编译、指令审计及GPU验收。当前production未切换。
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

## 验收与计时

- 合成输入覆盖全零、随机、饱和、正负交替、E2M1全部编码及零scale；新旧输出FP32逐位一致。
- 24真实样本重新计算对O0的FP64 reduction MSE；新旧同后端MSE必须完全相同。
- 同进程、相同输入、交错顺序、warmup50/repeats200/3轮；保留所有CV及离群值，不锁频。
- conversion-only独立批量摊销inner100；compute-only GEMM、cold及steady total均保留直接计时。
- 同一候选function必须有TMA、正确MMA语义，I2F消失且FADD/IADD存在；资源和spill检查保留。
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

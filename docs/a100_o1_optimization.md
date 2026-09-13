# A100 O1 优化与验收

状态：候选等待 A100 审计与性能验收，暂不替换旧 O1 默认实现。

目标是同一 A100、24 个相同输入的 O1 GEMM-only 延迟低于 cuBLASLt FP16 O0；
不保证通过削弱数值要求或选择性删除计时值达到目标。原 SM120/5090 后端保持独立。

## 不改变的语义

- E2M1 精确映射为 `2*E2M1` INT8 基值；A 仍为原始 prepared INT8。
- 每 K32 独立 INT32 partial，按原顺序执行
  `acc = fma(float(partial), A_scale * (decode(W_scale)*0.5), acc)`。
- partial 和最终 FP32 accumulator 留在寄存器；每个输出元素只写一次。
- 公共预处理不计时；独立转换批量摊销，cold/steady total 单次直接计时。
- 保留 `baseline` 作为同进程配对对照，不重采或重新量化 trace。

## 本轮候选

`csrc/sm80/o1_optimized.cuh` 独立实现 SM80 cp.async 双缓冲：

1. global→shared 的 16-byte copy 与 shared→register LDSM 共用 CuTe XOR swizzle 布局；
2. W scale 每 CTA/column/K32 仅解码一次，在 stage-local shared buffer 中向输出行复用；
3. row A scale 在 K 主循环外加载；不将其乘法移至最终输出，以维持原 FMA 递推；
4. K64/K128 stage 内分别处理 2/4 个 K32，不合并不同 scale 的 partial；
5. 比较固定 64×64、128×64、128×128 输出 tile，256线程/CTA；没有运行时 autotune。

后续有界候选增加 64×32 输出 tile、512线程/CTA 配置，并消除尾部冗余 CTA barrier：
下一轮开头 wait+barrier 已保护旧槽位全部 reader，下一次 prefetch 在该 barrier 后才覆盖。
所有候选仍需要 Compute Sanitizer 验证，不以源代码推理代替运行验收。

UE8M0/2 通过 FP32 位模式精确生成，包括 code 0/1 的 subnormal；不再调用通用 ldexpf。
scale 以每列连续2/4字节读取，shared store 按列合并，避免转置式写入的 bank 冲突。

带 `_exp` 的候选还将 `A_scale * 2^(code-128)` 改为指数位加法。计时前严格检查所有
A_scale 是正 normal，且全部可能乘积也为有限 normal；否则明确记录
`exponent_scale_fast_path=false` 并执行相同 tile 的正常 FP32 FMUL kernel。
正常范围内该操作逐位等价，不改变最后 FMA 或 group 顺序，不是把 A_scale 移到末尾。
此候选目的是将部分 scale 工作从 FP32 乘法换为整数加法，是否有收益由配对测试决定。

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

### NCU 定位的阶段性证据

不锁频 `baseline_full_ncu` 的旧 O1 Duration 约4.36ms，动态 warp 指令为1,429,766,144，
shared load bank conflicts 为150,994,944。首个 swizzle/shared-scale 候选 Duration约1.19ms，
动态指令427,556,864，DRAM throughput约7.56%，XU执行管线的 elapsed峰值占比约77.31%。
这里 NCU 指标用于定位原因，性能验收仍用普通 CUDA Event，不将 profiler Duration混入主表。

4096³的 K32 partial 转换数为 `4096*4096*128 = 2,147,483,648`，不是 Tensor Core MMA
指令数。常规类型转换的吞吐应与INT8 Tensor Core峰值分开分析；参见
[CUDA 12.8 Arithmetic Instructions](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#arithmetic-instructions)。

动态 shared-memory opt-in 在计时前完成。新候选要求 M/N/K 被相应 tile 整除；
非法候选/对齐条件直接报错，不伪装成自动 fallback。

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
python scripts/benchmark_a100_o1.py --validate --output runs/a100_o1_screen_v1
python scripts/audit_a100_o1.py --output reports/a100_o1_audit_v1
python -m pytest tests/unit/test_a100_o1_optimization_contract.py -q

# 对通过初筛的候选进行24样本配对。candidate_name 替换为实际候选。
python scripts/benchmark_a100_o1.py --samples 0 --rounds 10 --warmup 5 --repeats 20 \
  --impl baseline candidate_name --output runs/a100_o1_paired_new
python scripts/benchmark_a100_o1.py --samples 0 --rounds 1 --warmup 50 --repeats 200 \
  --all-modes --impl candidate_name --output runs/a100_o1_four_modes_new
```

脚本保存逐轮原始 timing、kernel 元数据、二进制 SHA、输入 manifest、运行环境与 GPU
状态快照。候选输出必须与旧 O1 逐元素完全一致，并重新计算 FP64 reduction 的 MSE vs O0。
所有 CV 异常保留，汇报 median/mean/P5/P95/CV，不把失败轮删除后标记通过。

审计必须在同一个实际候选 function 内确认 PTX cp.async+S8 MMA、SASS LDGSTS+IMMA，
并检查 local/stack/spill。profiling 不替代普通 CUDA Event 性能测量。

## 结果

待服务器验收后填写；旧 O1 约4.39 ms与 O0约0.603 ms只是历史基线，不与不同时间的候选
结果直接相除作为配对加速比。

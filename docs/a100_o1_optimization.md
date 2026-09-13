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

# A100 O1/O3 原生 INT4 对照实验

状态：SM80 后端已增加，等待 A100 实机验证与 24 样本结果。本文不预设 O3/O1 性能比。

## 范围与架构适配

O1 保持 E2M1 精确转 INT8 base，逐 K32 scale/FP32 累加；O3 保持 G128 MXFP4
转 Q4、INT8 激活拆成 low U4 与 high S4，分别进行 U4×S4、S4×S4 MMA，
合并 `low+16*high`，逐 G128 scale/FP32 累加。转换代码与 5090 共用。

A100 不具备 TMA。SM80 后端采用共同的 cp.async 双缓冲与 warp-cooperative
搬运/计算方式，O1 输出 tile=64×64、K stage=64，O3 输出 tile=64×64、K stage=128。
这属于 A100 上的成对移植实验，不能把与 5090 之间的差异全部归因于 INT4 指令。
O3 必须在同一个实际 kernel 的 SASS 中同时出现原生 U4×S4 和 S4×S4 IMMA。

## 服务器环境

全部项目、环境、缓存、临时文件均位于 `/home/zlouyang` 下。
系统已有 CUDA 12.8.93、驱动 570.124.06、A100 PCIe 40GB。
使用用户目录 Miniconda，Python 3.10、PyTorch 2.7.1 cu128、固定 CUTLASS 4.5.2。

```bash
export TMPDIR=/home/zlouyang/tmp
export PIP_CACHE_DIR=/home/zlouyang/.cache/pip
export PATH=/home/zlouyang/miniconda3/envs/adangel-a100/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=0
cd /home/zlouyang/ADAngel_oyzl
python -m pip install PyYAML==6.0.2 typing_extensions==4.12.2 numpy==2.1.3 ninja==1.11.1.3 setuptools==75.8.0 wheel==0.45.1 pytest==8.3.5
bash scripts/fetch_cutlass.sh
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps
```

编译生成 `adangel._sm80`；默认 `_sm120` 构建路径仍独立存在。

## 验证、审计、真实数据运行

```bash
python scripts/run_a100_experiment.py --validate-only --output reports/a100_validation
python scripts/audit_sm80.py --output reports/a100_audit
python scripts/run_a100_experiment.py \
  --data data/prepared/llama2_7b_prefill_o0_o4 --output runs/a100_o1_o3 \
  --warmup 50 --repeats 200 --inner 100
```

使用已有 24 个 prepared trace，逐文件校验 SHA-256；无需下载或加载大模型。
每个样本先用共用的 cuBLASLt O0 后端得到 FP16 输入、FP32 累加/输出参考。
O1/O3 交错测试四种模式，转换阶段摊销 100 次，GEMM 与端到端采用单次直接计时。
保存环境、编译 commit、输入 manifest、全部原始 timing、各阶段统计、FP64 reduction MSE。
CV≥3% 的记录保留并列出，不通过剔除数据制造稳定结果。

验收需同时具备：合成边界数据正确性、真实数据 finite FP32 输出与 MSE、
同入口 SASS 原生 INT4 证据、24 样本成对计时及对实现差异的说明。

参考：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)，
`mma` 的 SM80 sub-byte 形状与 `cp.async` 异步数据搬运说明。

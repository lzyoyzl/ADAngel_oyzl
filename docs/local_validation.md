# 本机验证记录

记录日期：2026-08-06。

## 当前开发机

- GPU：NVIDIA GeForce RTX 3050 Ti Laptop GPU，compute capability 8.6，4 GiB。
- WSL Python：3.10.12。
- CMake：3.30.5；Ninja 可用。
- `nvcc`：不可用。
- WSL PyTorch：未安装。

因此当前机器不能编译 `sm_120a`、运行 O2 指令，亦不能产生任何正式性能/MSE
结果。本项目没有用 RTX 3050 Ti 或 Python reference 冒充 RTX 5090 实验。
O0/O1 正式后端源码已经实现，但每次源码更新后的编译、数值和性能验收仍必须在目标
RTX 5090 上完成。

## 已完成验证

```text
22 项纯 Python 单元测试通过（含 O0/O1 正式后端静态契约与 fused tile ownership）
python/adangel、tests、scripts 全量 compileall 通过
scripts/fetch_cutlass.sh 与 scripts/audit_instructions.sh 的 bash -n 通过
doctor 正确报告：PyTorch is not installed
正式配置解析通过：4096^3、FP32 output、50 warmup、200 repeats、sm_120a
```

本轮没有安装驱动、CUDA 或 Python 依赖，也没有下载模型/CUTLASS。服务器配置仅以
`docs/server_dependencies.md` 和 `requirements/` 的锁定清单交付。

## RTX 5090 必做项

1. 按 `docs/server_dependencies.md` 人工配置服务器。
2. 运行只读的 `python scripts/check_server_prereqs.py --skip-cutlass`。
3. 主动运行 `scripts/fetch_cutlass.sh` 并核对完整 CUTLASS SHA。
4. 再次运行 `python scripts/check_server_prereqs.py`。
5. 构建扩展；确认 `capabilities()` 中 `o0_fp16_tc=true`、`o1_int8_tc=true`。
6. 运行 `python scripts/validate_o0.py`、`python scripts/validate_o1.py` 和对应原生集成
   测试，确认 O0 HMMA 与 O1 fused WMMA、转换、FP32 输出和四种计时模式全部通过。
7. 用 `4096^3` 运行 `validate_o1.py`，归档 `reports/o1_4096_fused_validation.json`。
8. 运行 `pytest tests/integration -q --run-sm120`，尤其是不同 A/B/K32 scale 的
   单 warp layout 验证。
9. 完成/启用 `csrc/sm120/o2_cutlass.cu`，数值验证通过后才可设置 O2 能力位。
10. 运行 `adangel doctor --require-native`，确认三个后端均可用。
11. 执行一次 SASS/PTX 指令审计并人工确认匹配到 O0/O1/O2 的正确 kernel symbol。
12. 先跑 1 样本 smoke，确认无 NaN/Inf、CV<3%，再跑完整 24 样本。

上述项目全部通过之前，`run_experiment` 按设计拒绝启动正式实验。

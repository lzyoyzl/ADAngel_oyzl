# O5–O10 编译、正确性与候选测量证据

此目录是原始 A100 输出的归档，不是 24 样本最终结果。没有过滤异常计时。

- CUDA 源码：`4507c6e`；运行脚本：`3a97ce2`。
- v3 二进制 SHA256：`3186f84b7c35c1933389a7ccafc092324d82a162aaf2eb5a9ddd20b8e42d0e3a0`。
- `reports/o5_o10_formats_v3/validation.json`：15 codec tests、120 定点转换、102 FP16、120 bitplane、408 Binary GEMM 等检查通过。
- `reports/o5_o10_memcheck_v3.*` / `racecheck_v3.*`：六种 Binary 配置、两种格式和两种 scale 布局，共 24 项；零错误/竞争。
- `reports/o5_o10_binary_audit_v3/audit.json`：同函数 BMMA+LDGSTS，无 INT8 代算；选定的自然 scale、64×128×256、256-thread 配置没有 LDL/STL 或栈 spill。
- `reports/o5_o10_int4_audit_v3/audit.json`：双 INT4 实际 SASS 审计；`allow-spills` 仅允许报告资源警告，不放宽 ISA 检查。
- `runs/o5_o10_horner_screen_v2` / `o5_o10_w16_screen_v3`：合成 4096³，3 轮，warmup=50/repeats=200。Horner 和 16-warp 没有可靠性能收益，不作为正式配置。

FP16 NCU 另在 `reports/ncu/o5_o10_v1/`：实际运行符号
`ampere_s16816gemm_fp16_128x128_ldg8_stages_32x5_tn`，SASS 含 `HMMA.16816.F32`。
其采集时是 v2 binary；精确环境与 hash 在 `runs/ncu_o5_fp16_v1/`。
FP16 GEMM 工厂和算法路径在 v3 未改变，v3 改动是源格式转换的精确解码。
NCU 重放耗时仅用于诊断，不混入普通 CUDA Event 结果。

首轮配置和 Binary NCU 见相邻目录 `a100_o5_o10_initial_v1`。
旧命名报告中的 O5/O6（双 INT4）对应现在的 O7/O8；本目录已使用命名版本 3。

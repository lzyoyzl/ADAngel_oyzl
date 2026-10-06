# v111 固定合作 CTA 编译证据

源 commit：`b79da2ae1aa660c6d392af95262149c7dad0d5c4`。本地实现先推 GitHub，
A100 核验 bundle SHA 后 fetch/ff-only；只在项目目录编译，不改环境或正式扩展。

仅一个 128×192×128 / 12-warp 候选，不做相邻尺寸扫描。Host-only 坐标检查、
控制机器码及原生双 INT4 审计通过，但补齐加权静态指令比 1.042020，预设门槛失败。
**无候选 GPU launch、真实性能、MSE 或 sanitize 数据，不是正式可运行后端。**

`reports/o378_roof_v111_cooperative_reuse_codegen/` 保存全部可读产物、源 hash、
命令、编译/映射日志、PTX、SASS、liveness 和失败 gate；顶层 `.log` 保存脚本输出。
`codegen.json` 中的 `SHARED:0` 是编译器对 static shared 的报告；本候选另需
59904 B dynamic shared，尚未做 runtime occupancy 查询。

完整含 cubin 和 host verifier 的归档存于本地/A100：
`tmp/o378_v111_codegen_complete.tar.gz`；SHA256：
`a991ae23cbeffa336b5165da5f8027da46a3409c6097610cf0d402ce92026f16`。
二进制不提交 Git，冻结测试核对其索引，并重放所有可读产物及源 SHA。

详见 [查重与停止依据](../../o7_o8_cooperative_reuse_20261007.md)。

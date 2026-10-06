# v95：SM80 四个现有 warp 的槽位握手（编译淘汰）

这是编译/资源证据，**没有启动 v95 目标 kernel**，不能作为性能、正确性、MSE或sanitizer验收。
本地实现先 GitHub 推送，再 A100 fetch/fast-forward；编译源码commit
`66189f28d302e994a799915d835c8d1228ad9349`，资源查询/门槛保护驱动commit
`24027d913b005733f169ce50ad2e7ec577153e4f`。

## 证据与结论

|指标|O3|O7/O8|
|---|---:|---:|
|旧 → 新整数循环静态指令|323 → 436|383 → 526|
|旧 → 新整数循环 local load|0 → 15|0 → 30|
|旧 → 新整数循环全 CTA barrier|1 → 0|1 → 0|
|每 G128 MMA / LDSM|64 / 16（不变）|64 / 16（不变）|
|新 entry 寄存器/线程|168|168|
|CUDA实际查询 CTA/SM|3|3|
|候选 local frame|72B|120B|
|ptxas spill stores / loads|104B / 112B|132B / 132B|
|预设 compile gate|失败|失败|

同entry native `IMMA.16864.U4.S4` / `.S4.S4` 与 cg copy通过；旧对照编码一致。
整数循环没有全 CTA barrier，但仍有 slot wait/release；同步不是消失，而是换成另一种协议。
静态指令计数包含分支/等待，不可当动态次数或退化百分比。明显新增热循环spill，不扩大GPU测试。
本地/A100相关单元测试均84通过；未重新编译正式扩展、未修改正式默认或5090。

## 可追溯文件

- `reports/o378_roof_v95_{o3,o78}_codegen/codegen.json`：源码/生成文件/二进制hash、命令、
  精确entry审计、编码旧对照、静态活跃范围和预设gate。
- 同目录 `*.ptx`、`*.sass`、`liveness.txt`、`resources.txt`、build日志与生成头文件：原始文本，字节保持不变。
- `reports/o378_v95_resources.json`：CUDA context中的CUfunction资源查询，不运行候选kernel。
- `reports/o378_v95_tests.log`：A100 84项相关测试通过。
- `reports/o378_v95_extension.sha256`：正式扩展仍为
  `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。

完整CUBIN等原始文件不提交Git，保存在本地和A100项目内
`tmp/o378_v95_compile_complete.tar.gz`，SHA-256：
`07ca62192630dd6a855250445544c21af856e9511c42a048d69dbcd8b8e6dd42`。
本目录只镜像文本；二进制hash由原始codegen和CUDA资源receipt绑定。

[实现、结果及既有最佳MSE](../../o3_o7_o8_slot_pipeline_20261006.md)。

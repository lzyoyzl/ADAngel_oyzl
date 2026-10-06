# v112 直接寄存器供数：冻结编译证据

源 commit：`d3537a05430a7efc8a823b42667eccd4550cdbcb`。先推 GitHub，A100核验bundle SHA后fetch/ff-only；
只在项目目录编译，未配置环境，未修改或重编译正式扩展。

仅一个固定候选。坐标/同entry双路原生INT4/旧控制编码审计通过；整数循环383→440条，
热local为52条，预设门槛失败。**从未启动候选GPU kernel，无新性能/MSE/sanitizer结论。**

`reports/o378_roof_v112_direct_fragment_codegen/` 保存原始可读产物、source/artifact SHA、
命令、编译日志、生成头、PTX、SASS、liveness、resource和失败gate；顶层log保存脚本输出。
原始源读取模型2.91×不是实际DRAM流量；128寄存器也不是runtime四CTA查询结果。

完整含cubin及host verifier的归档：`tmp/o378_v112_codegen_complete.tar.gz`，SHA256：
`b232f35a5a630c773be171b8ec16681f36ac67e791cdfc2fce106162c5bc2e59`。
二进制仅保留在本地和A100归档，不提交Git；测试重放可读证据和源SHA。

扩展前后SHA同为 `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
参见[机制查重与停止依据](../../o7_o8_direct_fragment_20261007.md)。

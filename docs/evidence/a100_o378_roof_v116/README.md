# v116：原生子块 unit 系数的数据门槛证据

只进行了 host CuTe 坐标校验、全24样本历史快照 SHA 核对及指令预算计算。
没有新 GEMM、GPU 启动、性能/MSE或安全验收结论。正式扩展 SHA 前后相同。

- O3-only：0275254，24条原始记录。保留最初结果，不用后来的 O7/O8 结论覆盖。
- All3：306f81d，72条记录。O7/O8只是 A 侧必要条件上界，不是完整 A/W 匹配。
- 原始编译日志、环境、坐标、逐样本与汇总结果全部冻结，11份文本 SHA 见 index.json。
- 两个 host 校验 ELF 留在服务器和完整归档中，不提交 Git；归档 SHA 见 index.json。
- 最初 coordinates.json 的 atoms_per_thread=8 指逻辑 N8 panel；All3明确区分
  每线程8个 N8 panel与16个原生 M16×N8 MMA atom。两次 O3 覆盖统计相同。
- 编译日志保留未引用的 device 声明警告，不据此声称启动过 GPU。
- A100原范围测试7项通过；离线证据回放见 tests/unit/test_roof_v116_evidence.py。

先成功推送各源码 commit 至 GitHub，再通过校验 SHA 的 Git bundle 在 A100 fetch/ff-only。
只写项目目录；原始快照、正式默认、源量化、转换实现与5090没有修改。

# O7/O8：全K激活factor面板（v127）

状态：独立候选，尚未A100编译/数值/性能验收；不改当前最佳、正式默认或5090。

## 瓶颈与新机制

当前v78已采用全K整数累加，不再逐G128做FP32转换。仍有两路INT4、整数factor乘加，以及metadata/payload的供数和同步。已有O8预热NCU见[瓶颈报告](o3_o7_o8_bottleneck_portability_20261007.md)，不能将PC采样占比当耗时比例。

新候选把CTA需要的32×64个**INT32激活factor**一次性加载进shared，始终按原G128索引读取；不把不同group的scale合并，也不省略加权。原每组Af cp.async/地址构造移到prologue。W factor、低/高INT4 payload仍用原两stage流水线，MMA八链、64×128×128 CTA、4 warp、guard/fallback和输出完全不变。

每线程4次16B copy覆盖32组×64行，加入第一批payload的commit；首次wait0和CTA barrier后读取，之后只读，不新增循环barrier。shared从34304增至41984B；需要查询实际驻留，不凭估计宣称occupancy不变。新增启动等待也计入GEMM，不能当免费预处理。

查重：不是v43寄存器scale提前加载、v81每stage metadata warp映射、v95 barrier替换、v105/v108 factor同值分组，亦不是v121循环内global只读访问。曾考虑全部A/W factor的16位压缩，但其解包/额外shared成本没有明确优势，因此**未实现、未测试**；本轮仅保留INT32 Af完整panel，没有压缩误差或新范围条件。

## 预设投入门槛

- 原v78控制完整编码不变；同一候选entry仍为两路原生INT4，非INT8替代。
- 热循环MMA 32+32、LDSM16、CTA barrier1不变；cp.async由10条变9条。
- 寄存器≤168、无热local、CUDA实际驻留保持3CTA/SM。
- 热循环静态指令至少减少3%；静态计数仅为投入筛查，不是速度预测。

通过才验证GPU坐标/同步、guard和逐位输出，再做24真实样本三轮1000/200/inner100配对；不做小规模性能筛选。正向后补四模式和MSE，负向或门槛失败则停止，不扫描panel窗口/压缩位数/邻近参数。

## 平台迁移边界

“低容量metadata可与大payload采用不同供数粒度”是可复用思路，但全K panel需权衡shared预算、启动开销和驻留数。A100用cp.async；其他平台必须重做copy、屏障、布局和资源审计，不能直接复制参数或声称已有跨平台收益。

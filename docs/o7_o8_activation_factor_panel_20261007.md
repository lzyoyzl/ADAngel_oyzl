# O7/O8：全K激活factor面板（v127）

状态：独立候选，A100编译及资源复核完成，尚未GPU数值/性能验收；不改当前最佳、正式默认或5090。

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

## 编译后独立资源复核（候选GPU执行之前）

编译commit `f8ab39d61c1a92d7c26d0e1d7e65dc7d2098f99d`。旧控制完整编码不变；热循环383→356条（−7.05%），allocated GPR168不变，实际驻留仍3 CTA/SM。shared34304→41984B；MMA32+32、LDSM16、barrier1不变，async copy10→9。

原“零热spill”门槛**保留失败**：新循环有1条32位LDL、0条STL，整个function local size为8B/线程。SASS `0xd10 STL[R1],R13` 在循环前保存，`0x10f0 LDL R14,[R1]` 每组读回，`0x1210` 与shared base相加后用于 `0x1240 LDSM[R97]`，是地址分量而非partial。并非没有代价：静态展开的未完成MMA链峰值8→6（不是硬件同时在飞计数），且prologue读取需要时间。

按用户此前允许少量spill的授权，**在任何候选GPU计时前**单独允许这一次地址reload，其他原门槛及3CTA保持要求不变；保存初始失败和独立复核，不将失败回写成通过。运行入口`benchmark_o78_activation_panel.py`会在启动候选前写出review receipt，先做不同row/group factor及非默认stream数值/同步检查，再直接24样本三轮1000/200配对。没有小规模性能筛选，也不扫描邻近panel大小/寄存器参数；若无收益即停止。

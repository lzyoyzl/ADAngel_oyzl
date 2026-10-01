# v46：G128输入地址的递增行偏移

状态：本地实现，待A100编译、审计和实测；未宣称收益。

v45删去的大量静态代码位于循环之外，核心G128循环没有明显简化。
本轮从原54/59出发，不叠加固定尺寸或v44的完整相对地址改写。
仅改变顺序prefetch的源地址生成：

```text
原：stage * M、stage * N → payload/scale源地址
新：cursor_a_rows = 0；cursor_w_rows = 0
    每发出一个G128预取后，分别加M和N
```

|方案|payload地址|scale地址|
|---|---|---|
|0|原54/59|原54/59，编码SASS对照必须逐字匹配|
|1|两条uint32_t递增行偏移|原按stage计算|
|2|同1|也使用对应行偏移；O3为UE8M0，O7/O8为FP32|

偏移不是截断后的指针，最终GPU指针仍64bit。
host保持正尺寸、M64/N128/K128对齐、M*K/N*K/M*N≤INT32_MAX及grid界限。
每次实际预取按0→G-1递增，包含2/3stage prologue和drain；最后一次更新仅生成一段尾后偏移，
不据此加载数据。K128模板约束保证每pipeline stage恰为一个G128，不合并各组scale。
新增CPU测试覆盖短K/奇数组数及接近现有地址上界，GPU再做bitwise与sanitizer验证。

CTA64×128×128、128线程/4计算warp、shared layout、预取提前量、wait/barrier、
两路原生INT4、scale与累加代码、最终FP32输出不变。
O3仍3stage，O7/O8仍2stage。没有新格式转换，也不重启magic-bias。

先核对同entry原生INT4、LDGSTS.BYPASS、资源与真实编码。
若候选与控制机器码相同，记录无实际改动；不能将源码乘法减少直接记作性能收益。
若编译得到不同kernel，执行有限GPU预检和memcheck/synccheck/racecheck，
再进行24样本×3轮、50warmup/200repeats的交错对照，保留CV失败与离群记录。
正向结果需独立确认及相应四模式验收，才考虑替换最佳；正式默认在此之前不变。
NCU核对动态地址指令、local/shared工作、eligible warp与等待，不能只追求零spill。

本轮独立cubin及Driver/Event接口，不重建正式扩展，不改RTX5090。
源码先在本地提交并推送GitHub，再由A100 fetch/ffmerge同步；所有A100写入限定项目目录。

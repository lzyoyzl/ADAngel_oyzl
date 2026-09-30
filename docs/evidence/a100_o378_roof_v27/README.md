# v27：静态环形流水线槽位（候选47/48，待GPU验收）

保持v24最佳41/42的CTA `64×128×128`、4 warp、B fragment跨M复用、
G128-major payload、两路原生INT4、寄存器partial、独立G128 scale、升序FP32 FMA。
47/48分别为两/三阶段，launch bound仍为128线程、3 CTA，转换路径不变。
正式默认与RTX5090代码不改。

## 唯一计划变动

原循环每个G128计算`slot = stage % stages`；候选每次展开2/3个物理槽位，
将槽位作为CuTe编译期整数传给同样的stage处理。三阶段预取的目标槽位也编译期确定。
所有G128仍按0→31顺序执行；不是split-K、树形求和、多累加链或合并数学scale。
每个展开phase保留尾部guard，支持K128/384/640等短/奇数G128。
wait_group、排空、CTA barrier及最终输出写回路径不变。

## 目的与代价

v26表明不能通过缩小tile来盲目追求零spill。本轮保留数据复用，只尝试减少
循环内取模、共享地址与槽位管理指令；不假设所有取模都对应昂贵整数除法。
编译器原先已能做强度削减，静态展开也可能增加代码体积、改变spill/指令缓存行为，
因此必须检查生成SASS、动态指令、正确性与真实配对性能，不能仅凭源码判断收益。

旧header完全保留；新TU使用隔离的device header。CPU契约核对除了槽位/循环调度外，
payload读取、MMA、scale、FMA和barrier源码不变，另模拟1–65组的槽位消费/覆写时序。
这不是GPU安全性证明，仍需A100合成、memcheck/synccheck/racecheck及24样本回归。

本地CUDA12.5独立TU编译已通过：六个实例均168寄存器/线程，存在8–32字节stack与spill，
不宣传为零spill；144项CPU/源码契约测试通过。这不是A100 CUDA12.8实测。

计划：GitHub→A100构建与旧SASS保持检查；
通过后同binary初筛、24样本MSE/性能，必要时NCU。所有负结果和CV失败保留。
本文件尚无性能结果，不计入当前最佳。

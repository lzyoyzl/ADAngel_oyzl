# v85：预排 MMA 寄存器布局，隔离供数候选

状态：首版编译通过但GPU正确性预检失败，已修正候选映射，等待重新验收；没有性能成绩，不改正式默认或5090。

首版错误发生于O7/random/candidate/conversion-only后的输出检查，尚未进入真实样本计时。
packing本身通过逐nibble oracle，但同一consumer warp的相邻N8 fragment在逻辑N上间隔16，
不是相邻8列。首版将后一fragment读成了另一N warp的列；因此只验证packing可逆不够。
修正W为N32、两个warp分别打包其自身两个N8 fragment；补充使用实际CuTe
`partition_A/B/C`逐坐标验证producer到consumer的主机程序，并检查它能识别旧映射错误。
原始失败的`reports/o378_roof_v85_codegen`和空screen目录保留，不覆盖；修正版使用新目录。

v80的等待样本仍集中在MMA消费点，v83/v84未改善速度。本候选不再扩大tile或stage，
而是把G128 packed payload无损排为每lane连续16B的MMA fragment，整数主循环改用
`ld.shared.v4.b32`，保留原cp.async、barrier、64×128×128、4warp、两阶段、八MMA链、
整数factor/guard和FP32 epilogue。这是供数假设，不意味着LDS必然比LDSM更快。

映射由pinned CUTLASS的CuTe `partition_A/B`构造；另有独立位排列oracle、完整payload与
输出对照，不能只凭公式或CPU槽位模型认定GPU正确。A每个M16×K64、W同一warp拥有的
两个N8×K64各产生32lane×16B，数据量不变；scale布局和数值完全不变。

初版显式增加A/W各一次重排kernel，并保留自然payload以供旧FP32安全回退。
4096³额外预分配24MiB，不在计时内申请；**重排不是免费离线工作**：
conversion-only与Cold计入W和A重排，steady只计A重排，compute-only提前准备。
复用v73原Event计时体，唯一新增执行是对应转换阶段的重排；正式默认不切换。
若GEMM没有确认收益，停止候选，不为负结果继续开发融合转换或扩大24样本。

```bash
python scripts/probe_o78_register_layout_codegen.py --output reports/o378_roof_v85_codegen_checked
python scripts/benchmark_o78_register_layout.py --cubins reports/o378_roof_v85_codegen_checked \
  --gpu-build reports/o378_roof_v85_codegen_checked --output runs/o378_roof_v85_screen_checked \
  --samples 4 --rounds 3 --warmup 50 --repeats 200
```

与v78同进程/同输入交错比较，原始Event、全部CV失败、输出MSE和指令身份都须保留。
只有正确性、原生INT4、同步/内存安全和初筛均通过，才决定是否进行24样本及四模式性能验证。

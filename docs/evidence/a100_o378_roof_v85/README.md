# v85：预排 MMA 寄存器布局，隔离供数候选

状态：已实现，等待A100编译/正确性/性能验收；没有新成绩，不改正式默认或5090。

v80的等待样本仍集中在MMA消费点，v83/v84未改善速度。本候选不再扩大tile或stage，
而是把G128 packed payload无损排为每lane连续16B的MMA fragment，整数主循环改用
`ld.shared.v4.b32`，保留原cp.async、barrier、64×128×128、4warp、两阶段、八MMA链、
整数factor/guard和FP32 epilogue。这是供数假设，不意味着LDS必然比LDSM更快。

映射由pinned CUTLASS的CuTe `partition_A/B`构造；另有独立位排列oracle、完整payload与
输出对照，不能只凭公式或CPU槽位模型认定GPU正确。A每个M16×K64、W相邻两个N8×K64
各产生32lane×16B，数据量不变；scale布局和数值完全不变。

初版显式增加A/W各一次重排kernel，并保留自然payload以供旧FP32安全回退。
4096³额外预分配24MiB，不在计时内申请；**重排不是免费离线工作**：
conversion-only与Cold计入W和A重排，steady只计A重排，compute-only提前准备。
复用v73原Event计时体，唯一新增执行是对应转换阶段的重排；正式默认不切换。
若GEMM没有确认收益，停止候选，不为负结果继续开发融合转换或扩大24样本。

```bash
python scripts/probe_o78_register_layout_codegen.py --output reports/o378_roof_v85_codegen
python scripts/benchmark_o78_register_layout.py --cubins reports/o378_roof_v85_codegen \
  --output runs/o378_roof_v85_screen --samples 4 --rounds 3 --warmup 50 --repeats 200
```

与v78同进程/同输入交错比较，原始Event、全部CV失败、输出MSE和指令身份都须保留。
只有正确性、原生INT4、同步/内存安全和初筛均通过，才决定是否进行24样本及四模式性能验证。

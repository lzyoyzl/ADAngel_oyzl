# v104：补齐 v98 尚缺的运行时证据，不伪称新优化

查重结论：v98 八 warp 全 K 方案已经实现、编译、审计，**未做过 GPU launch 或性能测试**。
本轮不重新实现、不重编译 CUDA、不扫描 tile/stage/warp，也不把它换名算作新优化。
复用原始 cubin，并核对原生成源码、原生双 INT4/cg copy、所有 artifact/source SHA。
旧 v98 文档与 10% 静态工作门槛保持原样，原 gate 仍 false。

此前用 +20.12% / +11.23% 的静态指令增量在预算下停止，不能证明真实吞吐更慢。
当前 v90 NCU 主基准的 issue active 为 O3 43.13%、O7 46.92%，
eligible warp/scheduler 为 0.65 / 0.75，说明不能仅按静态指令增量判定就绪度收益。
v98 预期 LDSM 工作 +50%；沿用旧资源模型，其必要 shared 容量约 0.258/0.274 ms，
仍低于目前主基准约 0.44/0.48 ms，但不是预测延迟。
因此只补一个尚不存在的运行时数据点，避免将启发式筛选当作真实实验结论。

首先用 CUDA Driver 查询原函数实际128/256线程、寄存器、local和active CTA：
若候选达不到2 CTA/SM、16 active warp/SM，停止，不启动性能测试。
若容量成立，直接24样本×三轮，单stream/50预热/200 Event，
同进程交错对比当前最佳与原v98，验证输出/MSE；没有小样本性能筛选。
所有离群值/CV保留。不凭静态计数推断收益。

O3/O7/O8 的量化、G128 scale、整数安全语义、两路原生 INT4 和 FP32 输出不改。
本轮 cached GEMM 的准备在计时外，两侧完全共用；不能当作转换或端到端收益。
若有显著收益，再单独补四模式和有限安全性；负结果不扩 NCU/四模式/参数扫描。
正式默认与SM80/SM120扩展不改。

只增加独立 host Driver，128/256线程共用相同 Event loop；不修改原 CUDA kernel。
先本地提交/GitHub成功推送，再 A100项目内 fetch、ff-only merge：

    python scripts/validate_unmeasured_eight_warp.py \
      --resources-only --output reports/o378_roof_v104_resources

本轮初始状态：待实际驻留查询，无新 GPU GEMM/MSE 或最佳结果。

首次资源查询在 Torch 仅初始化、尚未创建实际 current context 时停止，**没有 GEMM launch**。
独立 harness 增加一个上下文 anchor 分配，失败目录/日志保留；不修改任何 cubin。
修正后资源目录使用新名字，不覆盖失败记录。
容量成立时执行完整协议：

    python scripts/benchmark_unmeasured_eight_warp.py \
      --output runs/o378_roof_v104_full24

O3 cached 模式沿用已有 CPU 保守 column guard；候选/最佳共用完全相同 metadata。
O7/O8 共用原 v73 GPU 准备及安全 oracle，并要求原始 source 全字段 SHA 与 v99 一致。
这些准备全部在 Event 外，不是在线转换/Cold/steady 的结果。

首次完整运行在输入准备阶段因引用可选 roof-tune 的 packed 字段而停止，
没有候选 GEMM launch 或性能记录。改为从正式接口的自然顺序 payload 在计时外重排，
并与 Split/Q4 软件参考逐元素核对；O0 参考使用其专用 `benchmark_o0` 接口。
失败 `runs/o378_roof_v104_full24` 与日志保留，修正运行使用新目录。

## 完整结果：停止八 warp 路线，不替换当前最佳

原 v98 cubin 首次实际运行完成；这不是重复已经做过的 GPU 优化测试。
实际查询：最佳 168 regs/线程、3 CTA/SM、12 warp/SM；
候选 O3/O7-O8 为 121/128 regs、2 CTA/SM、16 warp/SM，候选 entry local 均为0。
CTA 输出仍64×128、K步长128，必要 MMA 工作不减，warp 加倍使 LDSM 工作增加50%。
旧编译 gate 仍 false，不回改历史判断。

24个真实样本×3轮×两实现×3后端，共432条记录，50次预热、200次单次 Event。
同进程交错顺序；共用输入、转换 payload、G128 scale、安全 metadata 及 Event loop。
转换和准备在 Event 外。没有新 NCU、conversion-only、Cold 或 steady-state 结果。

|后端|当前最佳同轮 GEMM median ms|八 warp median ms|配对吞吐变化|配对 speedup 95% CI|CV≥3%（最佳/候选，各72条）|
|---|---:|---:|---:|---|---:|
|O3|0.433152|0.480256|−10.72%|0.8875–0.8997|8/13|
|O7|0.463872|0.497664|−6.58%|0.9308–0.9384|13/12|
|O8|0.467968|0.497664|−6.79%|0.9270–0.9384|13/13|

median 延迟：先对每样本三轮取中位数，再取24样本中位数。
配对 speedup：每样本每轮计算最佳/候选，先取三轮中位数，再对24样本聚合/bootstrap；
因此不能直接用表内两个总体 median 的比值替代配对结果。
三个后端的24个样本配对 speedup 均小于1。
未锁频，采样时钟1125–1410MHz；所有原始值和 CV 失败保留，不宣称严格全阶段稳定通过。
本轮负收益方向一致，停止，不用反复重测或筛除离群值寻找正收益。

|输出 MSE（候选和最佳完全一致）|参考|Median|Mean|
|---|---|---:|---:|
|O3|O0|0.006653010287409885|0.007578847013302749|
|O7|O5|0.005536172273439442|0.005053635851002639|
|O8|O6|0.004411084910985704|0.004381379299073540|

12项合成正确性覆盖随机/零/饱和/fallback、非默认stream及group-tail，均通过；
432条真实输出均 finite FP32、与最佳逐位一致，MSE不变。
O7真实样本无fallback；O8最多12个fallback CTA，与两侧共同guard一致。
原生 U4/S4＋S4/S4 MMA、cg copy及旧编码通过原v98同entry审计，cubin SHA未变。
本轮不因负收益扩展 sanitizer 或四模式，不能宣称完成新的全尺寸内存安全验收。

**结论**：额外驻留不等于加速。该几何提高就绪warp数量，同时增加重复片段读取、
归一化指令工作，并降低驻留CTA数；实测净效果为负。
没有新NCU分解，不能把全部损失精确归因于某一类指令。
保留O3 v89、O7/O8 v78+v73主基准及v99微调候选，正式默认/扩展/5090均不变。
不再重做8-warp几何、相邻warp参数、tile/stage/cache或已有partial扫描。
后续候选必须在历史账本中证明机制未测过，并实际减少工作或改变关键依赖，而非只提高occupancy。

实现 commit `3db13b908dbe222a46d55b58f53a92c59dbf10e9`，GitHub推送后A100同步执行。
本地18项证据/契约测试通过；A100当时14项原契约测试通过。
[完整原始记录与复核入口](evidence/a100_o378_roof_v104/README.md)。

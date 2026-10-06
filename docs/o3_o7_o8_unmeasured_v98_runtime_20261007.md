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

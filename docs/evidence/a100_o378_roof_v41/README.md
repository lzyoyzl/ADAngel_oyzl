# v41：独立搬运 warp，负结果

**不采用。GEMM 最佳仍为 O3/54、O7/O8/59；转换最佳、正式默认和 RTX5090 不变。**

## 改动与资源

保留 CTA `64×128×128`、4个计算 warp（2×2）、每线程64个FP32 accumulator，
另加1个 producer warp搬运 packed A/W 和scale；消费者的MMA、G128顺序、scale与输出代码不变。
O3仍为3阶段、O7/O8仍为2阶段 `cp.async.cg`。
这是独立cubin实验，不接入正式调度。mode0的编码SASS与现有54/59逐条一致。

|资源|mode0 原控制|mode1 搬运warp|mode2 搬运warp、限制寄存器|
|---|---:|---:|---:|
|线程 / CTA|128|160|160|
|寄存器 / 线程|168|168|128|
|O3 stack / spill stores / loads B|16 / 12 / 12|0 / 0 / 0|112 / 304 / 320|
|O7/O8 stack / spill stores / loads B|8 / 8 / 8|0 / 0 / 0|56 / 116 / 136|
|Driver查询最大活跃CTA / SM|3|2|3|
|最大活跃计算warp / SM|12|8|12|

动态shared memory保持O3 50688 B、O7/O8 34304 B；新增warp并非免费资源。
mode1消除了spill，但减少了可驻留的计算warp；mode2恢复3个CTA，却产生明显spill。
spill表为ptxas编译报告，不是运行时总字节数。两候选均保持原生U4×S4和S4×S4，不是INT8替代。

同步采用每个stage的ready/free两个named barrier：producer等待槽位释放，
提交copy并等待本warp的copy完成，再发布ready；四个consumer等待ready，
完成所有shared读取后发布free。producer收尾等待最后几个槽位被消费。
使用显式160线程计数的 `barrier.cta.sync/arrive`，不加要求CTA一致到达的 `.aligned`。
该握手遵循[CUDA 12.8 PTX barrier定义](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar)。
测试覆盖小于、等于和超过stage数的group数量，但不宣称验证任意launch形状。

## 24样本配对结果

24样本×3后端×3轮×3配置，共648条compute-only记录；warmup50/repeats200。
同进程、同输入、原生Driver/Event计时，样本内循环换序。无锁频，不删除CV失败记录。

|后端|mode0 median ms|mode1 ms / 配对吞吐变化|mode2 ms / 配对吞吐变化|
|---|---:|---:|---:|
|O3|0.479744|0.537600 / **−10.69%**|0.698368 / **−31.32%**|
|O7|0.505088|0.567296 / **−10.91%**|0.603136 / **−16.67%**|
|O8|0.507904|0.569344 / **−10.89%**|0.609280 / **−16.60%**|

速度比先在同一样本内汇总三轮配对，再跨24样本汇总，不等于表中两个median直接相除。
mode1的描述性bootstrap 95%速度比区间依次为
[0.887619,0.895238]、[0.888889,0.892532]、[0.887701,0.893048]；
mode2为[0.685131,0.688596]、[0.831579,0.835034]、[0.832215,0.836975]。
均低于1，未发现收益。
CV≥3%的mode0/1/2记录分别为O3 25/15/11、O7 24/17/21、O8 25/16/17，每项分母72。
因此这不是全阶段严格稳定性验收；负结果不晋级conversion/cold/steady复测，不声称端到端提升。

|后端 / FP16参考|Median输出MSE|Mean输出MSE|相对原最佳|
|---|---:|---:|---|
|O3 / O0|0.006653010287410|0.007578847013303|逐位相同|
|O7 / O5|0.005536172426666|0.005053635833762|逐位相同|
|O8 / O6|0.004411084948645|0.004381379215302|逐位相同|

所有被检查输出为finite FP32；原始trace及prepared文件SHA、源格式重放与provenance随记录保存。

## NCU：搬运重叠收益没有抵消驻留和同步代价

mode0/mode1各采首个真实样本一次，`--set full --cache-control all --clock-control none`。
仅采O3/O7，不把O7 profiling当作O8实测；mode2的spill证据来自编译报告和配对性能，未采其NCU。
通过同entry symbol、128/160 block size及168寄存器核对捕获对象。
NCU duration只作诊断，不用作上表Event延迟。

|指标|O3 mode0→1|O7 mode0→1|
|---|---:|---:|
|NCU duration ms|0.398080→0.460480|0.422144→0.483648|
|动态warp指令|99.32M→99.53M|116.29M→115.66M|
|LDSM指令|4.19M→4.19M|4.19M→4.19M|
|BAR指令|0.262M→0.655M|0.262M→0.655M|
|Source shared wavefronts|29.62M→29.62M|31.29M→30.80M|
|Local theoretical sectors|3.18M→0|2.10M→0|
|Achieved occupancy（含producer）|18.00%→15.13%|17.99%→15.13%|
|Eligible warps / scheduler|0.626→0.473|0.775→0.527|
|Issue active|42.48%→36.13%|47.91%→40.27%|
|PC not-issued采样中barrier占比|7.68%→21.44%|9.43%→23.81%|
|1410MHz理想重叠容量下界 ms|0.220347→0.220347|0.220347→0.220347|

IMMA、I2F、FFMA均仍为每项16,777,216条；O3/O7的FMUL分别仍为524,288/16,777,216条。
搬运分工没有减少数学工作，也没有减少LDSM。
虽消除了local访问，候选每SM的计算warp上限从12减至8，ready/free握手增加，
可发射warp和发射利用率下降。上述证据共同支持“驻留与同步代价抵消搬运分离收益”，
但不是把总退化精确分摊给各因素的因果消融。
PC采样占比不是时间占比；总shared wavefronts不是HBM流量。

固定工作量的乐观下界仍由MMA/I2F容量约束，未缩小与该下界的差距。
**0.220347ms是假设理想重叠的必要容量下界，不是已经证明可以达到的kernel时间。**
v40/v41一起说明：不能只靠增加warp或消除spill来接近上界；应保留4个计算warp和复用，
优先减少实际供数/后处理工作，或在不增加CTA线程数与长期活跃寄存器的前提下改善短距离调度。
不继续盲目提高occupancy，也不重启magic-bias或已否定的大窗口树形累加。

## 验证、归档与复现

- 普通预检、racecheck、memcheck、synccheck分别完成180项检查。
  每次3后端×5种shape×4种pattern×3配置；K=128/256/384/640/4096，
  随机/零/极值/零scale，非默认stream，与当前最佳逐位一致，
  并满足FP64语义参考 `rtol=1e-3, atol=1e-3`。
- racecheck为0 hazards / 0 errors / 0 warnings；memcheck和synccheck均0 errors。
  这是所列几何和数据的有限检查，不代表任意并发场景。
- 六个entry通过原生INT4和异步copy审计；控制SASS与54/59完全一致。
- 归档前A100相关CPU测试270项通过；新增4项归档测试重算性能/MSE、ISA、资源/安全、NCU。
- 正式扩展未重建，SHA保持
  `fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。

编译、正确性、安全和主测提交：`2c12fc5093a77a3c863bce657616c78b79eb2419`。
NCU与270项CPU测试提交：`636375571a5f1afdb990ca2d7bf24715a791c8fa`（CUDA源码不变）。
均先在本地实现并推送GitHub，再同步至A100项目目录。

```bash
python scripts/probe_roof_producer_codegen.py --output reports/producer_recheck
python scripts/audit_roof_producer_probe.py --directory reports/producer_recheck \
  --best-sass docs/evidence/a100_o378_roof_v41/reports/o378_roof_v41/best_controls.sass \
  > reports/producer_recheck/audit.json
python scripts/validate_roof_producer_probe.py --cubins reports/producer_recheck \
  --output runs/producer_validation
python scripts/benchmark_roof_producer_probe.py --cubins reports/producer_recheck \
  --output runs/producer_trace24 --samples 24 --rounds 3 --warmup 50 --repeats 200
python scripts/profile_roof_producer_probe.py --directory reports/producer_recheck/ncu \
  --cubins reports/producer_recheck --runs-prefix runs/producer_ncu --candidate 1
python -m unittest discover -s tests/unit -p 'test_roof_v41_evidence.py' -v
```

仓库保存文本、SASS、原始Event样本和NCU导出；cubin、PTX、Driver .so及完整 `.ncu-rep`
留在A100项目的对应目录，不提交二进制。
传输归档SHA：`90d9c0302cbb286e215a06168701883dbb1483b24a49ff30b2ae577424bb12ca`。

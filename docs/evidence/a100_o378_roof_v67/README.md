# v67：O7/O8 全 K 整数累加，刷新 cached GEMM 候选

## 方法与范围

本轮得到用户明确授权，只测试独立候选，不修改正式默认或 RTX 5090 后端。
保留 tune59 的 CTA `64×128×128`、4 warp、2-stage cp.async、N64 fragment 流、
G128-major payload、两路原生 U4×S4 / S4×S4 MMA 与一次最终输出写回。

将源 scale 精确拆成 `整数 factor × 2^anchor`。每组仍使用自己的原 scale：

```text
P_g = dot(A_low,W) + 16*dot(A_high,W)
I += P_g * A_factor[row,g] * W_factor[col,g]   # INT32
Y = FP32(I) * A_base[row] * W_base[col]
```

O7 公共实数因子为 `4*T_W`，O8 为 `T_A/4`。这是代数等价的重新结合，
不保证保持旧 FP32 舍入位置，不是新的量化规则，也不使用 magic-bias。
CPU arbitrary-precision oracle 用 payload 平方和的 Cauchy 界保护每个整数乘积及所有前缀；
另保护两侧 factor 的乘积。失败 CTA 统一走旧 tune59 FP32 body，非法输入在 host 拒绝。
不满足充分安全界不等于实际发生过溢出。

**当前只测 cached compute-only。** CPU guard、factor/base 准备及 H2D 均在计时外；
每个 sample/variant 约 0.66–0.74 秒，单独记录在 `guard.preparation_wall_ms`。
它不是生产转换开销或 Cold/steady 结果，不能被隐藏或称为端到端最佳。
新增 metadata 1,089,536 B；实际在线准备尚未实现。

## 性能

先完成 4 样本×3轮筛选，O7/O8 配对吞吐 +6.81%/+8.30%；随后只确认这一候选。
以下为 **24 样本×3轮**，每轮 warmup50 / repeats200，同进程、同输入、同 native Driver Event
启动器，循环交错顺序，保留全部 288 条记录与 57,600 个原始 Event 时间。

|后端|同轮旧最佳59 ms|全K候选 ms|配对吞吐提升|描述性 bootstrap 95% CI|
|---|---:|---:|---:|---|
|O7|0.500736|0.468480|+6.84%|[1.065359,1.071895]|
|O8|0.501760|0.469248|+7.04%|[1.066362,1.073224]|

延迟是每样本跨三轮 median 后在24样本取 median；提升是配对 speedup 的 median，
不是两个总体 median 的简单相除。不能与之前不同 run 的 +18.47%/+18.51%直接相加，
也不能把旧24样本四模式中最快数字拼进这里。24个trace有关联，区间仅为描述性对照。

共享、未锁频 GPU：旧59/全K候选 CV≥3% 的条数 O7 为35/72、36/72，
O8 为35/72、31/72。不能声称严格稳定性门槛全部通过；原始失败记录均保留。

## MSE

|后端与参考|实现|Median 输出MSE|Mean 输出MSE|
|---|---|---:|---:|
|O7 / O5|旧59|0.005536172426666|0.005053635833762|
|O7 / O5|全K整数|0.005536172273439|0.005053635851003|
|O8 / O6|旧59|0.004411084948645|0.004381379215302|
|O8 / O6|全K整数|0.004411084910986|0.004381379299074|

全部24样本通过 `rtol=1e-5, atol=1e-12` 的 paired MSE 回归，
输出对旧59满足 `rtol=1e-3, atol=1e-3` 且 finite FP32。
O7/O8 新旧输出最大差 1.525879e-5 / 7.629395e-6，
逐样本输出差MSE最大 2.171332e-14 / 2.134541e-14。
control 与当前59逐位相同；全K候选不宣称逐位相同。

O7 的49,152个CTA全部通过整数安全界；O8只有 `layer_24_o_proj` 的12个CTA回退，
其余49,140个CTA走整数路径。已测这张混合回退矩阵的输出与 MSE。

## 审计与验证

- 同一正式候选 entry 的 PTX/SASS 均有 U4×S4、S4×S4 原生INT4 MMA；无INT8 MMA，
  cp.async 的 SASS 为 `LDGSTS.BYPASS`。不是靠 probe 或其他函数匹配。
- 两入口均168 registers/thread、34304 B dynamic shared、128 threads、2 stages；
  ptxas 各报告8 B stack、8 B spill stores及8 B spill loads。**不是零spill。**
- 24项 CPU 精确编码、guard、overflow及前缀证明测试通过；32项 GPU 合成检查覆盖
  不同行/列/group scale、全零、极值、随机、多CTA、零scale、forced fallback与非默认stream。
- 有限 memcheck、synccheck 各32项检查，均报告0 errors；未声称无限规模安全或已做racecheck。
- 单独24样本 payload proof 复核原生 packing/FP32 scales 与 guard 所用reference逐位相同；
  此次0预热/1次调用仅用于正确性，**不得把它的时间/CV作为性能结论**。
- `status=2` 在 host拒绝，本轮未直接执行 device invalid/no-store 分支。

CUDA源码提交 `dcff70e`，主要测量提交 `b174ded`；payload proof补充 `d6f327a`。
新增4项离线证据测试重算288条计时、配对结果、MSE、资源与指令；与24项metadata测试合计28项通过。
完整传输归档 SHA-256 为 `71a63474e9fefd180bb246b8e24c7aadfca02eaf0aa3df12205a0ba9760241ff`。
Git保留原始JSON/JSONL、PTX/SASS及日志；cubin与driver在A100和本地完整归档保留，不作为源码依赖。
正式 `_sm80.so` SHA-256 始终为
`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`，未重编译或切默认。
源码只在本地编辑、推送GitHub后在A100 fetch/fast-forward merge。

## 结论与下一步

当前合理结论是：**全K整数重排确实减少了 cached GEMM 的成本，约7%的收益得到24样本确认，
但尚未证明端到端收益。** 两路MMA、LDSM、tile、驻留规模均未减少，所以这不是接近纯MMA
容量下界的大幅突破；本轮没有新增 NCU 动态指令或硬件瓶颈占比结论。

后续重点是 GPU 在线 factor/guard，与激活转换融合或共享已有统计，权重准备缓存，
然后纳入 conversion/Cold/steady 四模式。若在线额外开销吃掉GEMM收益，则保留旧完整最佳。
不扫描更多同类参数，不切正式默认。

复测命令（fresh输出目录，A100既有环境）：

```bash
python scripts/probe_o78_fullk_codegen.py --output reports/fullk_o78_codegen_recheck
python -m pytest tests/unit/test_o78_fullk_integer_metadata.py -q
python scripts/benchmark_o78_fullk_integer_probe.py \
  --cubins reports/fullk_o78_codegen_recheck --output runs/fullk_o78_recheck \
  --samples 24 --rounds 3 --warmup 50 --repeats 200
```

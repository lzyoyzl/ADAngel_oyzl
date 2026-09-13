# A100 O3 优化：目标与验收记录

状态：首批候选开发/验收中，O3 production仍保留原baseline。
目标一未通过前，不开始目标二的5090 magic-bias部署。

## 目标

在同一A100、相同24份4096³真实输入上，让O3相对**当前优化后O1**的配对吞吐比
`T_O1 / T_O3 > 2`，即O3延迟低于O1的一半；不能拿旧O1约4.44ms作为目标分母。
验收不仅看一次最快值，还包括24样本配对、四模式、MSE、同function原生INT4指令审计、
无spill与Compute Sanitizer；保留全部原始计时及CV异常。

## 不变的数学语义

- 原INT8激活无损拆成低U4、高S4，`a=lo+16*hi`，不改变原激活量化。
- 权重仍来自FP16→MXFP4-G128→Q4；本轮不更换定点转换规则或数据。
- 使用U4×S4、S4×S4两个原生INT4 Tensor Core路径，不换成两个INT8 kernel。
- 每G128把两路INT32 partial重构为`low+16*high`，按原group顺序执行FP32 FMA。
- 原输出是唯一逐位回归参考；MSE仍对同一A100的FP32 O0输出计算。

## 首批候选

`csrc/sm80/o3_optimized.cuh`复用O1中有效的优化思路：

1. cp.async与LDSM共用swizzle，明确区分4bit nibble坐标和16-byte搬运坐标。
2. 每CTA/列/G128解码一次W scale，在shared memory中复用；A row scale循环外加载。
3. FP32位解码替代通用ldexp；正常范围guard下使用精确指数位加法。
4. partial留在寄存器；采用精确magic bias，INT8×Q4的G128保守界为
   `abs(partial)<=128*128*8=131072`，仍位于该转换的精确范围。
5. 双缓冲K128/K256 pipeline；K256内部保持两个独立G128 scale，不合并缩放。
6. 编译期CuTe坐标与最终单次输出store。

初筛候选：`64×64×128`、`128×64×128`、`128×64×256`，均256线程。
遇到非normal/极端scale时使用同tile的正常浮点转换路径，显式记录guard状态，不进入软件GEMM。

首轮单样本初筛（不是最终验收）：旧O3约1.858ms，64×64/K128候选约0.613ms，
同轮优化后的O1约1.010ms；O3/O1配对吞吐比约1.64，未达到2倍。三个候选逐位一致、
原生INT4指令及无spill审计通过。128×64/K256用了173register/thread，初筛没有优势。

`swizzle64_ncu`：Duration约606.46us，DRAM8.81%、L1/TEX63.58%、FMA43.0%；
No Eligible43.93%、理论occupancy37.5%。按每issue-active的平均warp stall比率，
barrier约1.914、wait约1.776、math-pipe约1.216、short-scoreboard约0.881，
long-scoreboard约0.295；这些是ratio，不是百分比。

因此继续测试64×32/64×128输出tile，以及CTA完整scale panel预取。后者限制K<=4096，
在**同一个被计时的GEMM kernel内部**加载、解码全部W scales，然后按G128顺序读取；
不跨调用缓存shared memory，也不把这部分处理挪到计时之外。

第二轮单样本64×128/K128约0.595ms，相对同轮O1约1.65倍，仍未达标。
第三轮完整scale panel缓存候选未优于该版本，不能据此切换默认后端。
审计发现缓存初始化中的运行时整数除法生成了`I2F.U32.RP`，而非partial转换退化；
后续将地址分配改为固定8线程/列，消除除法并覆盖小K非4整数倍group的安全加载。
新增`*_exp`候选仅使用指数位scale，保留普通I2F，单独检验magic bias对O3的收益；
O3每G128转换一次，而O1每K32转换一次，两者的转换吞吐压力不能直接类比。
上述均为待验收候选，不改变生产O3默认值。

## 命令

所有修改先在本地`/root/ADAngel_oyzl`完成并推送GitHub，A100仅同步、构建、运行。
服务器源码目录`/home/zlouyang/ADAngel_oyzl`，写入不得超出`/home/zlouyang`。

```bash
ADANGEL_BUILD_CUDA=1 ADANGEL_CUDA_TARGET=sm80 MAX_JOBS=4 \
  python -m pip install -v -e . --no-build-isolation --no-deps
python scripts/benchmark_a100_o3.py --validate --output runs/a100_o3_screen_new
python scripts/audit_a100_o1.py --variant o3 --output reports/a100_o3_audit_new
python -m pytest tests/unit/test_a100_o3_optimization_contract.py -q
```

目标二将单独记录SM120实现、独立构建和5090实测；A100结果不能替代5090验收。

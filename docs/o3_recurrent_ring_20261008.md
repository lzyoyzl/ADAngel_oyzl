# O3 三阶段槽位递推（v140，编译门槛停止）

**结论：不采用。** 虽然移除了两次取模，热循环静态指令反而增加3.72%，并出现新的local读写。
没有执行候选GPU、性能/MSE或NCU测试，不把静态增幅当作实测延迟变化。

## 目的与边界

继续针对 GEMM，不修改转换、源量化、G128 独立 scale、两路原生 INT4、全K整数安全 guard 或正式默认。
O7/O8 的两阶段 `%2` 没有同样的取模问题，本轮不迁移；5090不改。

v89 最佳 O3 的整数热循环仍两次计算三阶段槽位：`group%3` 和 `(group+2)%3`。
实际 SASS 中包含乘 `0xaaab`、移位、PRMT、乘3及减法，并非看到没有 `UIMAD.WIDE` 就能认定取模已经消失。
当前首要瓶颈仍是 MMA/数学管线与有限 eligible warp；本候选仅减少供数地址前的串行工作，不能承诺大幅收益。

新实现只维护一个 `slot=0→1→2→0` 的循环计数器：

```cpp
int slot = 0;
for (int group = 0; group < 32; ++group) {
    // 原 wait、barrier、payload/factor prefetch 和全部计算保持。
    const int future_slot = slot == 0 ? 2 : slot - 1;
    // group+2<32 时，下一预取写 future_slot。
    // 当前组读取 slot；两者从不相同。
    slot = slot == 2 ? 0 : slot + 1;
}
```

与 v27/v93 的区别：**不展开2/3个完整G128循环体、不扩大partial存活区间**，每次循环仍一个G128。
不是再次尝试相邻tile、warp数量、stage数量或预取位置。

## 预先规定的门槛

编译前固定：整数热循环静态指令至少减少3%，不增加分配寄存器、不新增热循环local读写；
同一候选入口保持32条S4×S4及32条U4×S4 MMA、16条LDSM、原copy/barrier数量，旧控制完整SASS编码不变。
减少静态指令不是实测加速。未达门槛则停止，不运行候选性能测试或扫描邻近计数器写法。

若通过，仍须实际资源、边界/安全性、逐位输出与MSE回归，再进行24样本×3轮配对。
保留所有CV与离群值；不做小规模性能初筛，未通过验收不替换当前最佳。

```bash
python -m pytest tests/unit/test_o3_ring_counter.py -q
python scripts/probe_o3_ring_counter_codegen.py --output reports/o378_roof_v140_codegen
```

## A100 编译审计结果

本地源码提交并推送GitHub后，A100项目内fetch/ff-only至
`a213dc08e0acc61427fc4449dc7555ee32d57fca`；固定CUDA12.8和原CUTLASS编译。
本地/A100的槽位、源代码等价和既有stage-cycle测试各16项通过。

| 同一整数G128热循环 | 当前最佳O3 v89 | 新v140 |
|---|---:|---:|
| 静态指令 | 323 | 335（+3.72%） |
| 分配寄存器/线程 | 168 | 168 |
| 静态活跃GPR峰值 | 166 | 166 |
| S4×S4 / U4×S4 MMA | 32 / 32 | 32 / 32 |
| LDSM / async copy / CTA barrier | 16 / 9 / 1 | 16 / 9 / 1 |
| 热循环local load / store | 0 / 0 | 4 / 1 |
| S2R | 2 | 4 |
| 未完成MMA链的静态峰值 | 8 | 8 |
| 全入口stack bytes/线程 | 16 | 32 |

旧控制完整entry机器字与冻结v89相同；同一候选入口PTX/SASS仍为两路原生INT4和异步copy，没有INT8替换。
两处`0xaaab`取模乘法确实消失，UPRMT从4条降为0；但新增状态改变了编译器的寄存器分配/寻址与调度，
出现更多S2R、移位/位操作和local读写，合计工作量没有减少。**源码里算得少不等于生成指令更少。**
这里的归因是生成代码层面的观察，未用运行时profiling量化任何一项的延迟影响。

原3%净减少及零热local门槛均失败，保留失败结论；不因用户允许少量spill就无条件测试——
本次目标本就是减少控制工作，既没有净减少，也没有增加MMA链并行度。
停止，不扫描邻近计数器表达式，不迁移O7/O8，不追加24样本性能/MSE/sanitizer。

当前最佳GEMM仍为O3 v89、O7/O8 v78，原正式扩展SHA不变；5090、源数据与转换均不改。
逼近有效吞吐上界的主目标尚未达成。15份原始文本及哈希见[证据目录](evidence/a100_o378_roof_v140/README.md)。

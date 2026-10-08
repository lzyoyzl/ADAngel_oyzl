# O3 三阶段槽位递推（v140，独立候选）

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

当前状态：源码与CPU契约测试待核对；尚无A100编译/性能结果。

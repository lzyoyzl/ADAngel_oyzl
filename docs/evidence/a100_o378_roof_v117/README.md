# v117：新的固定high×16路由，24样本负向

只改公开PTX的固定high重构，非v71/v87可变scale方案。编译门槛通过后直接完整24样本配对，
没有4样本性能初筛、参数枚举、正式扩展重编译或默认切换。

同一正式GEMM entry双原生U4/S4＋S4/S4、8条源级独立链、CTA64×128×128/128threads/2stage、
168regs/34304B shared、整数热spill0。旧v78控制编码完全一致。
固定high的21 IMAD.SHL＋43 SHF→64 SHF；全循环IMAD族158→137，总指令383→385。

24×2 variants×3 rounds×2 policies=288条，每条200个原始Event；warmup1000，单stream交错。
O7/O8配对吞吐变化−1.469%/−1.468%，置信区间均低于1，旧/新MSE完全相同、输出逐位一致。
64项synthetic与12项边界通过；少量CV≥3%全部保留。准备完全复用v73，正式扩展SHA未变。

`reports/`冻结全部编译文本、完整配对/源provenance/GPU快照/validation/原始Event与完成日志。
`index.json`记录每份文本SHA、源commit与完整归档SHA。
完整归档 `tmp/o378_v117_codegen_compute24_complete.tar.gz` 同时保存未提交Git的候选CUBIN，
SHA256=`ceebb1ef64fcfba0c0f53477ee708526ed6bd68e126c2121d7e90f55a2629ef0`。
编译commit=`c65bbf96ca86e5c0797929dfcae5d5976b659f6d`，运行commit=`509c1c58ba2dfa5a8f2bb0cc061b921704ff32e5`。

停止本路线，不试相近seed/opcode，不改O3、量化、转换、trace或5090。
没有新增conversion/Cold/steady-state成绩，也没有新的NCU或sanitizer验收结论。
正确性与性能结果不代表目标上界已经达到。

```bash
# 仅重算冻结证据，无GPU运行或编译
python -m pytest tests/unit/test_o78_high_funnel.py tests/unit/test_roof_v117_evidence.py -q
```

[查重、方法和完整结论](../../o7_o8_fixed_high_funnel_20261007.md)。

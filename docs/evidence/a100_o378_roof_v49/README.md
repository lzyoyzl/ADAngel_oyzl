# v49：B fragment lookahead 被编译器消解

**本轮没有新的加速结果。** 两个候选与原最佳54/59的完整编码SASS逐条一致，
因此停止性能晋级，保留原最佳。正式默认、扩展和RTX5090均未改变。

## 1. 测试内容

保持CTA64×128×128、四warp、N64流式片段，每路16个INT32 partial/线程。
只为下一N64片段增加B寄存器缓冲，当前片段消费完后直接使用该缓冲，不重复读取shared。

|方案|源码中的下一B片段读取位置|结果|
|---|---|---|
|0|原54/59按N片段读取|精确机器码控制|
|1|当前片段首个M atom计算前|编译为与0相同的机器码|
|2|当前片段首个M atom计算后|编译为与0相同的机器码|

v48扩大了B和partial窗口，本轮仅改变B供数安排，因此假设不同。
但源码安排并未成为实际机器码差异；不能据此声称获得了新的load/compute重叠。

## 2. 编译、审计与数值结果

|指标|O3，0/1/2|O7/O8，0/1/2|
|---|---|---|
|完整编码SASS与原最佳|全部相同|全部相同|
|静态指令数|944/944/944|880/880/880|
|寄存器/线程|168/168/168|168/168/168|
|shared bytes|50688，均不变|34304，均不变|
|stack bytes|16/16/16|8/8/8|
|spill store/load bytes|12/12，三者均相同|8/8，三者均相同|
|原生U4×S4、S4×S4及LDGSTS.BYPASS|同entry确认|同entry确认|

编码检查比较实际SASS指令字，不仅比较指令数量。p0另与原54/59比较，避免用被改动的控制自证。
依据为`codegen.json`、`audit.json`、三份probe SASS及`best_controls.sass`。

本地和A100各354项相关CPU测试通过。GPU预检180项：
3后端×3方案×5形状×4pattern，K128/256/384/640/4096；
随机/全零/INT极值/零scale、不同row/column/group scale、非默认stream。
全部finite FP32、逐位等于54/59，对FP64语义参考满足rtol=atol=1e-3。
K4096的预检形状为64×128×4096，不冒充本轮完整4096³验收。

本轮未重跑24真实样本MSE、配对性能、四种计时模式、NCU和sanitizer：
候选没有产生新执行指令，没有必要把同机器码的计时噪声当作优化效果。
下面只列保留的**历史最佳**，不当作v49新数据。

|后端 / 参考|历史最佳GEMM median ms|历史Median MSE|历史Mean MSE|
|---|---:|---:|---:|
|O3 / O0，v30候选54|0.477952|0.006653010287410|0.007578847013303|
|O7 / O5，v33候选59|0.503808|0.005536172426666|0.005053635833762|
|O8 / O6，v33候选59|0.506368|0.004411084948645|0.004381379215302|

参见[当前最佳的独立配对口径](../../o3_o7_o8_current_best.md)。不将上述跨run绝对值相除。

## 3. Insight及下一步边界

源码中“提前读取”不等于机器码中“更早读取”。本次重排被编译器规范化，
说明应先核对实际依赖/供数工作变化，再投入大规模性能测量。
本轮不能证明B供数已达到最优，也不能证明手写SASS一定有益；不进行不受支持的机器码修改。
MMA/I2F等工作不变，原理想重叠容量下界约0.220347ms仍不是已达到或保证可达的延迟。
跨G128整数对齐累加待确认，不实施；magic保持停止。

## 4. 来源与复现

源码`425538371efd91777060cfa6f6de5a0b5ab8e065`，先本地实现/push，再A100 fetch/ffmerge。
正式扩展SHA仍`fe9c2b18d89787231bf988b704a29d2041edcfdad558c98acee3f5b2195b366f`。
文本归档SHA`2d8e5b30a029d0d2e59eee7705fd2837539c09979dcbae1ef747296c2dee8a2b`。
二进制/PTX/坐标可执行文件保留在A100项目内，此处归档文本证据。

```bash
python scripts/probe_roof_b_lookahead_codegen.py --output reports/b_lookahead_recheck
python scripts/audit_roof_b_lookahead_probe.py --directory reports/b_lookahead_recheck \
  --best-sass docs/evidence/a100_o378_roof_v49/reports/o378_roof_v49/best_controls.sass \
  > reports/b_lookahead_recheck/audit.json
python scripts/validate_roof_b_lookahead_probe.py --cubins reports/b_lookahead_recheck \
  --output runs/b_lookahead_validation
python -m unittest discover -s tests/unit -p 'test_roof_v49_evidence.py' -v
```

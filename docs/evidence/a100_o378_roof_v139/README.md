# v139 O7 联合转换验收：完整证据

本轮仅组合既有v118权重转换和v106激活lookup，两侧使用同一个v78原生两路INT4 GEMM。
对照为v118 W＋原v73 A；候选为v118 W＋v106 A。
复用v126已冻结库的control，不执行其失败的packed-MX8候选，不改变旧编译门槛。

24样本×3轮四模式，1000预热、200次测量、转换inner100。
576条记录、345600个Event值，全部保留；转换total配对吞吐+1.09199%，steady+0.19416%。
Cold的+0.18383%区间触及1，未确认；GEMM同码、0新增收益。
输出逐位一致，vsO5 MSE median/mean为0.005536172273439442/0.005053635851002639。

## 文件与范围

- `runs/o378_v139_full24/`：每个Event、四模式结果、环境、72次GPU快照、24个源数据identity、数值验证。
- `runs/o378_v139_preflight/`：32项合成、9项边界和131072个保留decoder组合。
- `runs/o378_v139_memcheck/`、`runs/o378_v139_synccheck/`及对应日志：定向row-conversion、小M/N K4096安全检查。不是全GEMM安全验收。
- `reports/o378_v139_memcheck.log`：首次CLI过滤语法错误，候选尚未执行；修正后的memcheck日志以`_r2`结尾。
- `reports/o378_v139_analysis.json`：服务器端复算；`analysis.json`：本地逐Event复算并增加SASS依赖与安全验证。
- `index.json`：所有归档文本SHA、分析SHA及引用的既有代码证据SHA。

原生扩展运行前后SHA均为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
运行commit为`f3cd64b86651e652b8492dd2dee3628441d5bf7a`。
转换库沿用v126编译commit `436e974e355d3c3e85eea32603db925546ff8109`；
库SHA `906dde016ff9516edda4f1ee682932dd5816f0706c66c698ea1e0c846cc837fe`。
原v78 GEMM cubin SHA `08badf82c4538c2d3695ca0d53fa4082669db12038f356892bd5ec85d9d9c11a`。
完整机器码/源文件已在v73/v78/v106/v118/v126证据中，不重复提交二进制。

归档`tmp/o378_v139_complete.tar.gz`的服务器/本地SHA一致：
`726b24efbe5e39776b2f3b944971bdde827c1b879244359b43f459bdd6e92759`。
Cold-W双方53/72、58/72条CV≥3%，全部保留，不能宣称各分项均稳定。

```bash
python scripts/analyze_o7_conversion_combo.py \
  --input docs/evidence/a100_o378_roof_v139/runs/o378_v139_full24 \
  --output tmp/v139_recomputed_new.json
python -m pytest tests/unit/test_o7_conversion_combo.py \
  tests/unit/test_roof_v139_evidence.py -q
```

分析输出必须为新文件。正式默认、O3/O8/5090实现均未修改。
GEMM主目标未达到；不追加微小查表邻近变体。

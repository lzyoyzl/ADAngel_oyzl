# v118：packed NVFP4 权重转换的完整证据

编译源码 `0891b5647cb77959b04d5ceb1802c72941853289`，运行源码
`0f006fc5f7c277fe08de3a9d3fc3ebc19f78564f`。本地提交并推送后，A100项目内
fetch/ff-only，未改正式扩展或5090。原扩展SHA和全部24份文本SHA见index.json。

24样本×3轮×4模式×2policy，共576记录、345600个原始Event值，无过滤。
新旧准备路径使用同一个v78 GEMM CUfunction；输入源格式identity逐字段与v99一致。
GPU全编码word检查131072项、合成32项、边界9项通过；全输出逐位相同，MSE未变。

新权重转换entry：392→320静态指令、31→30寄存器、256B shared、一个barrier，
stack/local=0、10次精确平方和POPC。全部旧入口编码SASS一致。
候选只减少转换工作，不是GEMM或量化语义改变。

权重conversion-only配对吞吐+21.82%，conversion total+6.42%，Cold total+0.65%。
GEMM/steady无收益。Cold W隔离批量阶段有57/72条CV略超3%，所有原值保留；
不声称严格所有阶段稳定验收通过。完整阶段统计与bootstrap在analysis.json。

memcheck与synccheck仅过滤两个新的转换entry，有限small-M/N、K4096与word检查
均为0 errors；不是完整GEMM/4096³/racecheck验收。未新增NCU。

完整包含独立库的原始归档在本地和A100项目tmp/o378_v118_complete.tar.gz；
SHA为a6998cbb1bc574d38b86c58559fe47ecbeb5699353bac7c7e11a3dae3f13ac71。
原始binary不提交Git，文本24份完整保留，路径/哈希与归档一一对应。

CPU复算（不用GPU、Torch或模型）：

```bash
python scripts/analyze_nv4_swar.py --input docs/evidence/a100_o378_roof_v118/runs/o378_v118_full24 --output tmp/v118_recomputed.json
python -m pytest tests/unit/test_nv4_swar_codegen.py tests/unit/test_roof_v118_evidence.py -q
```

保留独立转换候选，不修改正式默认，不扫描相邻布尔或查表变体。
GEMM接近有效吞吐上界的目标尚未完成，最佳GEMM仍为O3 v89、O7/O8 v78。

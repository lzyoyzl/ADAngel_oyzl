# v104：原 v98 的缺失运行验证

不是新kernel或重复GPU优化。复用原v98 cubin，旧静态gate false及所有旧证据保持原样。
原v98只编译/审计；本轮第一次实际测量该kernel，直接24样本，不做小样本性能初筛。

- 运行源码 commit：`3db13b908dbe222a46d55b58f53a92c59dbf10e9`。
  本地GitHub推送成功，再A100项目内bundle fetch及ff-only同步，未重编CUDA。
- 最终接口修正bundle SHA-256：
  `b65541c8c721855295dae0bc03e69484971d14ec0c26fd44665905ef2801def0`。
- 服务器/本地原始归档 `tmp/o378_v104_complete.tar.gz` SHA-256：
  `b1f2aa52af5c6c95e743838b7846fdae0777cd2fbfb84eb37e4893089455ca7a`。
- 正式SM80扩展未改：
  `94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
  正式默认与5090不改；所有服务器写入都在 `/home/zlouyang/ADAngel_oyzl`。

原始文件：

- [432条单次Event样本、全部CV及逐样本MSE](runs/o378_roof_v104_full24_r2/results.jsonl)。
- [六行汇总及配对bootstrap](runs/o378_roof_v104_full24_r2/summary.json)。
- [12项正确性检查](runs/o378_roof_v104_full24_r2/validation.json)。
- [源码/原cubin/guard/扩展/trace身份](runs/o378_roof_v104_full24_r2/environment.json)。
- [48个source全字段身份](runs/o378_roof_v104_full24_r2/source_provenance.jsonl)：与v99逐字段SHA一致。
- [未锁频GPU采样](runs/o378_roof_v104_full24_r2/gpu_snapshots.jsonl)：1125–1410MHz。
- [实际资源查询](reports/o378_roof_v104_resources_r2/build/resources.json)：两候选16warp/SM、local0。
- [完整成功日志](reports/o378_roof_v104_full24_r2.log)及[A10014项契约测试](reports/o378_roof_v104_unit_tests.log)。
- [首次上下文失败](reports/o378_roof_v104_resources.log)及[首次payload字段失败](reports/o378_roof_v104_full24.log)：
  均在候选launch前停止，未覆盖日志，不算性能记录。

主机编译日志和各build资源receipt保留；独立host `.so`仅在本地归档，不提交GitHub。
同entry双INT4/cg copy/原编码证据链接原[v98](../a100_o378_roof_v98/README.md)，不重复粘贴或修改。

候选O3/O7/O8配对吞吐−10.72%/−6.58%/−6.79%，三者24样本均更慢；
432条输出均逐位一致且MSE不变。停止八warp路线，最佳仍O3 v89、O7/O8 v78+v73。
这是cached GEMM结果，准备在Event外，两侧共用；没有新增转换/Cold/steady/NCU或sanitizer验收。
不筛除CV失败，不把绝对时钟漂移拼成新最佳。全部结论可用CPU证据测试复核：

    python -m pytest tests/unit/test_unmeasured_eight_warp.py \
      tests/unit/test_eight_warp_fullk_probe.py tests/unit/test_roof_v98_evidence.py \
      tests/unit/test_roof_v104_evidence.py -q

本地18项通过（CPU平台CV末位浮点差用1e-12容差，不改原始数据）。
详细[结果、MSE与查重停止依据](../../o3_o7_o8_unmeasured_v98_runtime_20261007.md)。

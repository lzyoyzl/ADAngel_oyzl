# v140：O3 recurrent ring 编译审计

本地commit `a213dc08e0acc61427fc4449dc7555ee32d57fca`先成功推送GitHub，
A100在`/home/zlouyang/ADAngel_oyzl`通过bundle fetch/ff-only同步后编译。
source bundle SHA：`cffcd2c9c21f299f04aa1d329e330b2bbde1921eb2763131aabb7a9c6e84d98c`。

15份原始文本（10,084,036字节）及SHA见[index.json](index.json)：生成头文件、PTX、SASS、
寄存器live-range、资源、编译命令/日志。原始归档`tmp/o378_v140_compile_complete.tar.gz`还保留CUBIN，
SHA为`ed29830f6ac87a24a8ca94e4166e6ad26b12319183a38f749060b72fdfe8851a`；不提交二进制。

结果：323→335条整数热循环静态指令、168 allocated/166 peak GPR，新增4条local load及1条store；
原生32+32 INT4、16 LDSM、9 async copy、1 CTA barrier保持。原v89控制完整编码一致。
两处取模被消除，但没有净工作量收益；原预设门槛失败，因此停止。

**没有执行候选GPU，没有新的性能、MSE、NCU或sanitizer结果。** 不能称为实测变慢3.72%或0%加速。
正式扩展SHA仍为`94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462`。
正式默认、既有最佳与5090均不改。

```bash
python -m pytest tests/unit/test_o3_ring_counter.py tests/unit/test_roof_v140_evidence.py -q
```

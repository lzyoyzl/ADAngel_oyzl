# SM120 MXFP4 microscale layout

O2 使用：

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4
.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
```

一个 warp 的逻辑 scale tile 是 `SFA[16,2]` 与 `SFB[2,8]`。两个 K32 scale
对应一条 k64 MMA。项目采用 PTX 文档的 canonical 映射，并把四个 selector 都设为 0：

```text
scale-a-data:
  lane = 4*q     : byte 0/1 = SFA[q,0],   SFA[q,1]
  lane = 4*q + 1 : byte 0/1 = SFA[q+8,0], SFA[q+8,1]
  lane = 4*q + 2 : ignored
  lane = 4*q + 3 : ignored

scale-b-data:
  lane = 4*q     : byte 0/1 = SFB[0,q], SFB[1,q]
  lane = 4*q + 1 : ignored
  lane = 4*q + 2 : ignored
  lane = 4*q + 3 : ignored
q = 0..7
```

A/B 是两个独立 `.b32` metadata operand，不能把它们拼成同一个 word。selector
`{byte-id,thread-id}={0,0}` 对 A 选择 quad 内 lower thread-pair，对 B 选择 quad 内
lane 0，并从各自 operand 选择低两个字节。

`tests/integration/test_sm120_layout.py` 为每个位置使用不同 scale，专门检测 lane、
byte、A/B 方向及转置错误。该 microkernel 只负责 layout/语义验证；正式 4096³
性能入口是 `csrc/sm120/o2_cutlass.cu` 的 tiled CUTLASS kernel。

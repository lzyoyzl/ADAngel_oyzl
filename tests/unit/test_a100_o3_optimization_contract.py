from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[2]


def test_o3_exact_bias_entire_conservative_g128_bound():
    for v in range(-131072,131073):
        biased=struct.unpack('<f',struct.pack('<I',0x4b400000+v))[0]
        out=struct.unpack('<f',struct.pack('<f',biased-12582912.0))[0]
        assert out==float(v)


def test_o3_twos_complement_split_is_lossless():
    for a in range(-128,128):
        u=a&255;lo=u&15;hi=u>>4
        if hi>=8:hi-=16
        assert lo+16*hi==a
        for w in range(-8,8):assert lo*w+16*(hi*w)==a*w


def test_o3_keeps_native_int4_and_g128_recurrence():
    s=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
    for expected in ['SM80_16x8x64_S32U4S4S32_TN','SM80_16x8x64_S32S4S4S32_TN',
                     'Groups=K/128','low(i)+16*high(i)','__fmaf_rn(value,scale,acc(i))',
                     'NibbleLayout','ByteLayout','cp.async.wait_group']:
        assert expected in s
    assert 'SM80_16x8x32_S32S8S8S32_TN' not in s


def test_cached_scale_panel_partition_covers_small_and_full_k():
    for n in (32,64,128):
        for groups in range(1,33):
            seen=[]
            for tid in range(256):
                first=(tid%8)*4
                for col in range(tid//8,n,32):
                    for j in range(4):
                        if first+j<groups: seen.append((col,first+j))
            assert len(seen)==n*groups
            assert set(seen)=={(c,g) for c in range(n) for g in range(groups)}


def test_magic_audit_checks_final_template_boolean():
    import re
    for fast in (0,1):
        for cached in (0,1):
            for magic in (0,1):
                for wn in (2,4):
                    for merge in (0,1):
                        symbol=f'_Z25adangel_sm80_o3_swizzledILi64ELi128ELi128ELb{fast}ELb{cached}ELb{magic}ELi{wn}ELb{merge}EEEvPKh'
                        for name in ('swizzled','swizzled_bound2'):
                            candidate=symbol.replace('swizzled',name)
                            match=re.search(r'swizzled(?:_bound2)?ILi\d+ELi\d+ELi\d+ELb([01])ELb([01])ELb([01])ELi[24]ELb([01])E',candidate)
                            assert (match[3]=='1')==bool(magic)


def test_g128_integer_reconstruction_can_reuse_one_partial():
    import random
    rng=random.Random(20260914)
    for pattern in range(1000):
        a=[rng.randrange(-128,128) for _ in range(128)]
        w=[rng.randrange(-8,8) for _ in range(128)]
        if pattern<4:
            a=[(-128,127)[pattern%2]]*128
            w=[(-8,7)[pattern//2]]*128
        high=[v//16 for v in a];low=[v&15 for v in a]
        partial=16*sum(x*y for x,y in zip(high,w))
        for start in (0,64):
            partial+=sum(low[i]*w[i] for i in range(start,start+64))
            assert -(2**31)<=partial<2**31
        assert partial==sum(x*y for x,y in zip(a,w))


def test_static_copy_exactly_covers_each_shared_tile():
    for rows in (32,64,128):
        for k in (128,256):
            for threads in (256,512):
                step=threads*16;size=rows*(k//2)
                offsets=[t*16+i*step for t in range(threads)
                         for i in range((size+step-1)//step) if t*16+i*step<size]
                assert sorted(offsets)==list(range(0,size,16))


def test_phase_pair_keeps_group_order_and_handles_odd_stage_count():
    for count in range(1,33):
        schedule=[]
        for stage in range(0,count,2):
            schedule.append((stage,0))
            if stage+1<count:schedule.append((stage+1,1))
        assert schedule==[(stage,stage%2) for stage in range(count)]


def test_cached_codes_swizzle_is_bijective_aligned_and_bank_distributed():
    def address(col,group):
        offset=col*32+group
        return offset^((offset>>5)&28)  # CuTe Swizzle<3,2,5>
    for n in (32,64,128):
        offsets=[address(c,g) for c in range(n) for g in range(32)]
        assert sorted(offsets)==list(range(n*32))
        for c in range(n):
            for g in range(0,32,4):
                assert address(c,g)%4==0
                assert [address(c,g+j) for j in range(4)]==list(range(address(c,g),address(c,g)+4))
        for g in range(32):
            for start in range(0,n,32):
                assert len({(address(c,g)//4)%32 for c in range(start,start+32)})==32


def test_o3_atom_stream_keeps_both_integer_paths_and_ordered_fma():
    s=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
    stream=s.split('if constexpr(Stream) {',1)[1].split('} else {\n      cute::clear(low);',1)[0]
    for expected in ('ra1','rh1','br1','SliceN=WN*(NarrowB?8:16)','pl(vi)+16*ph(vi)',
                     'SM80_16x8x64_S32U4S4S32_TN','SM80_16x8x64_S32S4S4S32_TN',
                     '__fmaf_rn(value,scale,acc(vi,mi,full_ni))'):
        assert expected in stream


def test_bound2_uses_separate_entry_without_changing_default_launch_bounds():
    s=(ROOT/'csrc/sm80/o3_optimized.cuh').read_text()
    assert '__launch_bounds__(32*(M==32?2:4)*WN) void adangel_sm80_o3_swizzled(' in s
    assert '__launch_bounds__(256,2) void adangel_sm80_o3_swizzled_bound2(' in s
    assert 'o3_body<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream>(a,w,as,ws,y,m,n,k)' in s
    assert 'o3_body<M,N,K,Fast,Cached,Magic,WN,Merge,StaticCopy,PhasePair,Stream,true>(a,w,as,ws,y,m,n,k)' in s
    assert 'SM75_U32x2_LDSM_N' in s
    assert 'o1_static_for<0,C::Groups>(process_group)' in s


def test_compact_packed_offsets_match_linear_layout_with_capacity_guard():
    for m,n,k in ((64,128,256),(128,128,512),(4096,4096,4096)):
        for row in (0,m//2,m-1):
            for byte in (0,k//4,k//2-16):
                lo=row*(k//2)+byte;hi=lo+m*(k//2)
                assert 0<=lo<hi and hi+16<=m*k<=0xffffffff
        for row in (0,n//2,n-1):
            assert row*(k//2)+(k//2-16)+16<=n*k//2<=0xffffffff
    host=(ROOT/'csrc/sm80/o1_o3.cu').read_text()
    assert 'a.numel()<=0xffffffffLL&&w.numel()<=0xffffffffLL' in host

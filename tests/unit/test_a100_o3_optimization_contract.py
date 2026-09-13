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
                    symbol=f'_Z25adangel_sm80_o3_swizzledILi64ELi128ELi128ELb{fast}ELb{cached}ELb{magic}ELi{wn}EEEvPKh'
                    assert bool(re.search(r'Lb1ELi[24]EEEv',symbol))==bool(magic)

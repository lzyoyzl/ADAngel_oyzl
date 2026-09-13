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

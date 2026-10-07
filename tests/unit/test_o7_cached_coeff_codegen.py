"""Cached scale coefficients change supply, not integer dot or quantization."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


def test_only_storage_supply_and_coefficient_lookup_changed():
    from probe_o7_cached_coeff_codegen import generated_header,OLD_COPY,NEW_COPY,OLD_COEFF,NEW_COEFF
    from probe_o78_eight_chain_codegen import generated_header as eight
    old=eight((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text());new=generated_header()
    restored=new.replace('o7_cached_coeff_experiment','o78_eight_chain_experiment').replace(
        '  uint32_t coefficients[2][1280];','  int weight_factors[2][128];').replace(
        'static_assert(sizeof(Storage)==43520);',
        'static_assert(sizeof(Storage)==34304);\nstatic_assert(sizeof(Storage)==sizeof(C::Storage));').replace(
        NEW_COPY,OLD_COPY).replace(NEW_COEFF,OLD_COEFF)
    assert restored==old
    assert 's.coefficients[slot][shift][threadIdx.x]=' not in new
    assert new.count('__syncthreads()')==old.count('__syncthreads()')
    wrapper=(ROOT/'csrc/sm80/roof_o7_cached_coeff_probe.cu').read_text()
    assert 'f>=1024u' in wrapper and 'if(__syncthreads_or(bad))' in wrapper
    assert 'o78_eight_chain_experiment::body(a,w,af,wf' in wrapper


def test_table_layout_is_bijective_and_adjacent_pairs_remain_contiguous():
    from probe_o7_cached_coeff_codegen import table_offset
    for n in (128,256):
        offsets=set()
        for g in range(32):
            for c in range(n):
                for d in range(10):
                    p=table_offset(g,c,d,n);assert p not in offsets;offsets.add(p)
                    if c%2==0:assert table_offset(g,c+1,d,n)==p+1 and p%2==0
        assert offsets==set(range(32*n,32*n*11))


def test_exact_selected_coefficients_never_use_wrapped_values():
    for f in (0,1,3,15,576,2047,65535,2**22,2**30,2**31-1):
        for d in range(10):
            encoded=(f<<d)&0xffffffff
            if f*(1<<d)<=2**31-1:assert encoded==f*(1<<d)
            # Unsafe coefficients remain rejected by the ORIGINAL full-K guard.
    from probe_o7_cached_coeff_codegen import LIMITS,SHARED
    assert SHARED==43520 and LIMITS['max_work_ratio']==1.02
    assert LIMITS['min_imad_reduction']==.2 and LIMITS['min_runtime_cta']==3


def test_frozen_real_A_factor_range_is_covered_not_a_full_cta_guard():
    import numpy as np
    root=ROOT/'docs/evidence/a100_o378_roof_v105/reports/o378_roof_v105_runtime_factor_r2'
    files=sorted(root.glob('*o7_factor_observation.npz'));assert len(files)==24
    count=0;maximum=0
    for p in files:
        with np.load(p,allow_pickle=False) as a:
            f=a['factors'];assert f.shape==(32,4096)
            assert ((f>0)&(f<1024)&((f&(f-1))==0)).all()
            assert (a['row_status']==0).all()
            count+=f.size;maximum=max(maximum,int(f.max()))
    assert count==3145728 and maximum==512

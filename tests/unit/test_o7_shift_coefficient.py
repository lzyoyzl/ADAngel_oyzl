"""v87 coefficient shift is exact, does not shift partial, and stays isolated."""
from pathlib import Path
import random
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o7_shift_coefficient_codegen import generated_header, eight_header, ROW_SHIFTS, NEW_COEFFICIENT, OLD_COEFFICIENT, MARKER


def test_all_legal_shift_counts_and_integer_boundaries():
    rng=random.Random(20261003)
    checked=0
    for d in range(31):
        limit=(2**31-1)//(1<<d)
        values={0,1,limit,max(0,limit-1)}|{rng.randrange(limit+1) for _ in range(512)}
        for weight_factor in values:
            coefficient=(weight_factor << (d & 31)) & 0xffffffff
            assert coefficient==weight_factor*(1<<d)<=2**31-1
            bound=min(131072,(2**31-1)//coefficient) if coefficient else 131072
            for partial in {-bound,-1,0,1,bound}:
                if abs(partial)>bound: continue
                product=partial*coefficient
                for acc in (0,-product,2**31-1-max(0,product),-2**31-min(0,product)):
                    assert -2**31<=acc<=2**31-1
                    assert -2**31<=acc+product<=2**31-1
                    assert acc+product==acc+partial*(1<<d)*weight_factor
                    checked+=1
    assert checked>100000


def test_only_coefficient_and_four_row_exponents_change():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    got=generated_header(source)
    assert got.count('shf.l.wrap.b32')==1
    assert got.count('row_shifts(ri,mi)=31-__clz(factor)')==1
    assert 'acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;' in got
    restored=got.replace('o7_shift_coefficient_experiment','o78_eight_chain_experiment').replace(
        ROW_SHIFTS,'').replace(NEW_COEFFICIENT,OLD_COEFFICIENT)
    assert restored==eight_header(source)
    assert got.count('cp.async.wait_group 0')==got.count('__syncthreads()')==1
    with pytest.raises(ValueError):
        generated_header(source.replace('// Same N64 stream and four independent MMA atoms as tune59.',
                                        '// unexpected operand stream boundary'))
    assert MARKER in got


def test_wrapper_retains_guard_and_old_fp32_fallback_no_production_dispatch():
    wrapper=(ROOT/'csrc/sm80/roof_o7_shift_coefficient_probe.cu').read_text()
    assert '#include "roof_o78_eight_chain_probe.cu"' in wrapper
    assert 'if(flag>1u) return;' in wrapper and 'if(flag==1u)' in wrapper
    assert 'O78::o3_body<64,128,128' in wrapper
    assert 'o7_shift_coefficient_experiment::body' in wrapper
    for path in (ROOT/'python/adangel/backends').glob('*.py'):
        assert 'shift_coefficient' not in path.read_text()

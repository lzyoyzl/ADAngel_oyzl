from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_o3_eight_chain_codegen import generated_header,async_header,BEGIN,END,LAMBDA_BEGIN,LAMBDA_END


def test_o3_ring_copy_scale_and_epilogue_preserved():
    source=(ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()
    old=async_header(source)
    new=generated_header(source).replace('o3_eight_chain_experiment','o3_fullk_integer_experiment')
    assert new.split(LAMBDA_END,1)[0]==old.split(LAMBDA_BEGIN,1)[0]
    assert new.split('  auto final_value=',1)[1]==old.split('  auto final_value=',1)[1]
    assert old.split(LAMBDA_END,1)[1].split(BEGIN,1)[0]==new.split(LAMBDA_END,1)[1].split('// Eight independent chains:',1)[0].rstrip()+'\n'
    assert new.count('cute::copy(')==old.count('cute::copy(')==6
    assert new.count('cute::gemm(')==old.count('cute::gemm(')==4
    assert 's.factor[slot][cute::get<1>(coord)]' in new
    assert 'activation_factors' not in new and 'weight_factors' not in new
    assert 'cute::_4{},cute::_2{},cute::_4{}' in new
    assert 'partial(i)*=16' in new and 'integer_group' not in new


def test_o3_source_boundary_changes_fail_closed():
    source=(ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()
    for marker in (BEGIN,END,LAMBDA_BEGIN,LAMBDA_END):
        with pytest.raises(ValueError):generated_header(source.replace(marker,'unexpected change'))

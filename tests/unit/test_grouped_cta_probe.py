from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_grouped_cta_codegen import mapping, generated_headers
from probe_o3_eight_chain_codegen import generated_header as o3_header
from probe_o78_eight_chain_codegen import generated_header as o78_header


@pytest.mark.parametrize('columns',[1,2,3,7,8,9,16,32])
@pytest.mark.parametrize('rows',[1,2,7,8,9,16,17,64])
def test_exact_coverage_and_guard_coordinates(columns,rows):
    sequence=[mapping(x,y,columns,rows) for y in range(rows) for x in range(columns)]
    assert len(set(sequence))==rows*columns
    assert set(sequence)=={(x,y) for y in range(rows) for x in range(columns)}
    for linear,(x,y) in enumerate(sequence):
        span=8*columns;first=(linear//span)*8;size=min(8,rows-first);within=linear%span
        assert (x,y)==(within//size,first+within%size)


@pytest.mark.parametrize('kind',['o3','o78'])
def test_only_cta_coordinates_change_in_math_and_fallback(kind):
    actual,fallback=generated_headers(kind)
    original=(o3_header((ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text()) if kind=='o3'
        else o78_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()))
    def undo(value,new,old):
        return value.replace(new,old).replace('roof_grouped_cta::tile().x','blockIdx.x').replace(
            'roof_grouped_cta::tile().y','blockIdx.y')
    assert undo(actual,kind+'_grouped_cta_experiment',kind+'_eight_chain_experiment')==original
    file,old=('o3_row_scale_epilogue_candidate.cuh','o3_row_scale_epilogue_experiment') if kind=='o3' else (
        'o78_unsigned_payload_candidate.cuh','o78_unsigned_payload_experiment')
    assert undo(fallback,kind+'_grouped_fallback',old)==(ROOT/'csrc/sm80'/file).read_text()
    assert 'blockIdx.' not in actual and 'blockIdx.' not in fallback


def test_no_preparation_new_work_or_scope_expansion():
    from benchmark_o78_grouped_cta import timing_contract
    from benchmark_o78_eight_chain_probe import timing_contract as previous
    for mode in ('conversion_only','compute_only','cold','steady_state'):
        new=timing_contract(mode,100)
        assert new.pop('cta_order_group_m')==8 and new.pop('new_preparation_or_layout') is False
        new['comparison']=previous(mode,100)['comparison']
        assert new==previous(mode,100)
    source=(ROOT/'csrc/sm80/roof_o78_grouped_cta_probe.cu').read_text()
    assert 'status[tile.y*(n/128)+tile.x]' in source
    assert 'o78_grouped_fallback::o3_body' in source

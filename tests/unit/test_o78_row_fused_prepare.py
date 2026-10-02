"""v73 row-fusion numerical code identity and complete CTA mapping."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))


def test_vector_quantization_and_packing_are_unchanged():
    old=(ROOT/'csrc/sm80/roof_vector_norm_conversion.cuh').read_text()
    new=(ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    for start,end in (('  unsigned words[','  // One subwarp'),
                      ('    if constexpr(Format==Kind::Hif4) value=','    effective[dst_group]=value;')):
        assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
    assert '__syncthreads();\n  if(threadIdx.x<32)' in new
    assert 'sat_term(square_by_group[t],unsigned(factor))' in new
    assert 'group_squares[src_group]=square;' in new
    assert 'decode<ScaleKind>(code_by_group[t]' in new
    assert 'newexp>223' in new and 'factor_bad?Cap:norm' in new


def test_one_row_256_threads_covers_32_groups_exactly():
    elements=[]
    group_leaders=[]
    for thread in range(256):
        group,lane=divmod(thread,8)
        elements.extend(group*128+lane*16+i for i in range(16))
        if lane==0:group_leaders.append(group)
    assert sorted(elements)==list(range(4096))
    assert group_leaders==list(range(32))


def test_host_keeps_v69_control_and_identical_timing():
    old=(ROOT/'csrc/sm80/roof_o78_fused_prepare.cu').read_text()
    new=(ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    old=old[old.index('extern "C" int roof_o78_fused_benchmark'):]
    new_tail=new[new.index('extern "C" int roof_o78_row_fused_benchmark'):]
    assert old.strip()==new_tail.replace('roof_o78_row_fused_benchmark','roof_o78_fused_benchmark').replace('RowFusedOnline x(','FusedOnline x(').strip()
    assert 'FusedOnline::weight(true)' in new and 'FusedOnline::activation(true)' in new
    assert new.count('adangel_o78_prepare_cta_guard<<<')==1
    assert 'factor_metadata<' not in new


def test_runtime_switches_preparation_but_not_gemm(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import benchmark_o78_row_fused as probe
    calls=[]
    def call(*args):
        calls.append(args)
        for i in range(len(args[-1])):args[-1][i]=1.0
        return 0
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=7))))
    driver=object.__new__(probe.Driver)
    driver.handles={0:11,1:11,2:11}
    driver.lib=SimpleNamespace(roof_o78_gpu_benchmark=call)
    case=SimpleNamespace(variant='o8',oracle={'status_flat':np.array([0])},a_source=1,w_source=2,
        state_pointers=3,m=64,n=128,a_multiplier=.25,w_multiplier=1.0,state={'y':'output'})
    for policy in (0,1,2):driver.run(case,policy,'compute_only',5,2,100)
    assert [c[0] for c in calls]==[11,11,11]
    assert [c[2] for c in calls]==[0,1,0]


def test_timing_metadata_distinguishes_fused_candidate():
    import benchmark_o78_row_fused as probe
    for mode in probe.base.MODES:
        control=probe.timing_contract(mode,100,0)
        candidate=probe.timing_contract(mode,100,1)
        assert control['preparation_implementation']=='fused_conversion_group_squares_then_metadata'
        assert candidate['preparation_implementation']=='row_fused_conversion_factor_metadata'
        assert control['group_squares_read_for_metadata'] and not candidate['group_squares_read_for_metadata']
        assert (control['weight_preparation_launches'],candidate['weight_preparation_launches'])==(2,1)
        assert (control['activation_preparation_launches'],candidate['activation_preparation_launches'])==(3,2)
        for key,value in probe.base.timing_contract(mode,100).items():
            assert control[key]==candidate[key]==value
        assert control['gemm_cufunction_identical_between_policies']
        assert candidate['gemm_cufunction_identical_between_policies']

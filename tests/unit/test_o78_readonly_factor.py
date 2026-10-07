"""Factor-only supply boundaries and predeclared compiler potential gate."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_o78_readonly_factor_codegen as probe
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_only_metadata_copy_and_reads_change():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    body=probe.generated_header(source).replace(
        'o78_readonly_factor_experiment','o78_eight_chain_experiment')
    assert body.replace(probe.COPY_NEW,probe.COPY_OLD).replace(
        probe.LOAD_NEW,probe.LOAD_OLD)==eight_header(source)
    assert body.count('__ldg(')==2 and body.count('__syncthreads();')==1
    assert 'static_assert(sizeof(Storage)==34304);' in body
    assert 'acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;' in body
    assert 'Groups=32;' in body and body.count('cute::gemm(LA{}')==2
    assert body.count('cute::gemm(HA{}')==2 and 'shared_partial' not in body


def test_drift_does_not_silently_produce_a_different_experiment():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    with pytest.raises(ValueError,match='factor-copy/read boundary'):
        probe.generated_header(source.replace('const unsigned first=threadIdx.x*4;',
            'const unsigned first=threadIdx.x*8;'))


def live(work=365,registers=168):
    return dict(allocated_gpr=registers,loops=[dict(kind='integer',static_instructions=work,
        opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
            'LDSM.16.M88.4':16,'LDGSTS.E.BYPASS.128':8,'BAR.SYNC':1,'LDG.E.CONSTANT':20})])


def test_gate_rejects_copy_reduction_without_meaningful_work_or_resource_improvement():
    old=live(383);new=live()
    assert probe.cost_gate(old,new)['passed']
    assert not probe.cost_gate(old,live(379))['passed']
    assert not probe.cost_gate(old,live(registers=176))['passed']
    for op in ('LDL.64','STL','LDS'):
        bad=live();bad['loops'][0]['opcode_counts'][op+'.U32']=1
        assert not probe.cost_gate(old,bad)['passed']
    bad=live();bad['loops'][0]['opcode_counts']['LDGSTS.E.BYPASS.128']=10
    assert not probe.cost_gate(old,bad)['passed']


def test_fullk_guard_fallback_and_output_abi_not_changed():
    source=(ROOT/'csrc/sm80/roof_o78_readonly_factor_probe.cu').read_text()
    old=(ROOT/'csrc/sm80/roof_o78_eight_chain_probe.cu').read_text()
    start='  const uint32_t flag=status[blockIdx.y*(n/128)+blockIdx.x];'
    end='  else\n'
    assert source.split(start)[1].split(end)[0]==old.split(start)[1].split(end)[0]
    assert '__launch_bounds__(128,3)' in source and 'float* y,int m,int n,int k)' in source
    assert probe.LIMITS['min_static_reduction']==.03
    # Every formal factor access fits the existing group-major panels.
    for group in (0,31):
        for block,extent,tile in ((63,4096,64),(31,4096,128)):
            indices=[group*extent+block*tile+local for local in range(tile)]
            assert 0<=min(indices)<=max(indices)<32*extent

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


def test_runtime_both_policies_keep_guard_and_restore_control(monkeypatch):
    import benchmark_o3_eight_chain_probe as runtime
    driver=runtime.Pipeline.__new__(runtime.Pipeline)
    control,candidate=object(),object()
    driver.device=control;driver.eight_chain=candidate
    calls=[]
    def fake(self,policy,*args,**kwargs):
        calls.append((policy,self.device))
        return {'kernel':{'weight_conversion_kernels':2}}
    monkeypatch.setattr(runtime.FullPipeline,'run_four',fake)
    for policy,symbol in ((0,runtime.CONTROL),(1,runtime.CANDIDATE)):
        result=driver.run_four(policy,'compute_only')
        assert result['kernel']['weight_conversion_kernels']==2
        assert result['kernel']['kernel_symbol']==symbol
        assert driver.device is control
    assert calls==[(1,control),(1,candidate)]
    with pytest.raises(ValueError):driver.run_four(2,'compute_only')
    def error(*args,**kwargs):raise RuntimeError('expected test failure')
    monkeypatch.setattr(runtime.FullPipeline,'run_four',error)
    with pytest.raises(RuntimeError):driver.run_four(1,'cold')
    assert driver.device is control


def test_runtime_summary_requires_complete_pairs_and_bitwise_outputs(monkeypatch):
    import benchmark_o3_eight_chain_probe as runtime
    rows=[dict(sample_id='x',round=0,mode='compute_only',implementation=i,
               bitwise_equal_current_best=True,payload_bitwise=True) for i in (0,1)]
    monkeypatch.setattr(runtime,'base_summary',lambda rows:'complete')
    assert runtime.summarize(rows,['x'],1,['compute_only'])=='complete'
    for broken,ids,rounds,modes in ((rows[:1],['x'],1,['compute_only']),
            (rows+rows[:1],['x'],1,['compute_only']),
            (rows,['x','y'],1,['compute_only']),(rows,['x'],2,['compute_only']),
            (rows,['x'],1,['compute_only','cold'])):
        with pytest.raises(ValueError):runtime.summarize(broken,ids,rounds,modes)
    rows[1]['bitwise_equal_current_best']=False
    with pytest.raises(ValueError):runtime.summarize(rows,['x'],1,['compute_only'])

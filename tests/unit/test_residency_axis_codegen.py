"""v107 source/compile gates; not GPU numerical/performance acceptance."""
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_residency_axis_codegen as probe
from probe_o78_eight_chain_codegen import generated_header as eight_header


def test_quantization_guard_copy_and_epilogue_unchanged():
    old=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    new=probe.generated_header().replace('o78_residency_axis_experiment','o78_eight_chain_experiment')
    assert old[:old.index('  auto tile_a=[&]')]==new[:new.index('  // Same warp ownership')]
    tail='  // Reuse the 64 INT32'
    assert old[old.index(tail):]==new[new.index(tail):]
    pipeline='  constexpr int Groups=32;'
    finish='    cute::copy('
    a=old[old.index(pipeline):];b=new[new.index(pipeline):]
    assert b.startswith(a[:a.index(finish)])


def test_fixed_partial_and_operand_budget_not_N128_window_repeat():
    text=probe.generated_header()
    assert 'cute::make_shape(cute::_4{},cute::_8{})' in text
    assert 'cute::make_shape(cute::_4{},cute::_2{},cute::_8{})' not in text
    assert 'cute::_32{},cute::_64{}' in text and 'cute::_128{},cute::_64{}' in text
    assert 'cute::Tile<cute::_32,cute::_128,cute::_64>' in text
    assert 'atr.partition_fragment_A' in text and 'aht.partition_fragment_A' in text
    assert 'auto atr=slice_mma.get_slice(threadIdx.x);' in text
    assert 'auto aht=slice_high_mma.get_slice(threadIdx.x);' in text
    assert 'cute::make_tiled_copy_A(LCopy{},slice_mma)' in text
    assert 'o1_static_for<0,2>([&](auto mb)' in text
    assert text.count('cute::gemm(HA{}')==text.count('cute::gemm(LA{}')==2
    assert text.count('bc.partition_S(tile_b(slot')==2
    assert 'const auto coord=coords(vi,mb,ni);' in text
    assert 'acc(vi,mb,ni)+=partial(vi,ni)*coefficient;' in text


def test_no_mandatory_work_increase_and_predeclared_potential_gate():
    ops={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16}
    old=dict(allocated_gpr=168,loops=[dict(kind='integer',static_instructions=383,max_live_gpr=166,opcode_counts=ops)])
    new=copy.deepcopy(old)
    assert not probe.worth_runtime(new,old)
    new['loops'][0]['static_instructions']=363
    assert probe.worth_runtime(new,old)
    new['loops'][0]['opcode_counts']['LDSM.16.M88.4']=24
    assert not probe.worth_runtime(new,old)
    new['loops'][0]['opcode_counts']['LDSM.16.M88.4']=16
    new['loops'][0]['opcode_counts']['LDL.64']=1
    assert not probe.worth_runtime(new,old)
    new=copy.deepcopy(old);new['loops'][0]['max_live_gpr']=150
    assert probe.worth_runtime(new,old)
    new['allocated_gpr']=176
    assert not probe.worth_runtime(new,old)


def test_kernel_retains_uniform_guard_and_existing_fallback():
    src=(ROOT/'csrc/sm80/roof_o78_residency_axis_probe.cu').read_text()
    assert 'if(flag>1u) return;' in src and 'if(flag==1u)' in src
    assert 'O78::o3_body<64,128,128' in src
    assert 'o78_residency_axis_experiment::body(a,w,af,wf,base_a,base_w,y,m,n,k);' in src
    assert '__launch_bounds__(128,3)' in src


def test_coordinate_proof_covers_source_halves_rows_columns_and_all_owners():
    src=(ROOT/'tests/cuda/validate_residency_axis_coordinates.cu').read_text()
    assert 'auto full_a=thr.partition_A(ia);' in src
    assert 'auto full_b=thr.partition_B(ib);' in src
    assert 'auto sa=athr.partition_A(ta);' in src
    assert 'cute::Tile<cute::_32,cute::_128,cute::_64>' in src
    for text in ('full_a(v,mb,half)','full_b(v,ni,half)',
                 'for(int count:owners) assert(count==1);','gpu_execution\\\":false'):
        assert text in src
    assert 'cudaLaunch' not in src

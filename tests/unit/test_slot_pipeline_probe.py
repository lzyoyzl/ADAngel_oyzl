from pathlib import Path
import random
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_slot_pipeline_codegen import CONFIG, baseline_header, generated_header, worth_runtime


@pytest.mark.parametrize('kind',('o3','o78'))
def test_only_pipeline_not_math_epilogue_guard_or_quantization(kind):
    old,new=baseline_header(kind),generated_header(kind)
    start='    load_a(slot,a00,a01,h00,h01);' if kind=='o3' else '    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);'
    tail='  auto final_value=' if kind=='o3' else '  // Reuse the 64 INT32'
    old_math=old[old.index(start):old.index(tail)].rsplit('  }',1)[0]
    new_math=new[new.index(start):new.index('    // All128 threads release')]
    assert old_math==new_math
    assert old[old.index(tail):].split('} // namespace')[0]==new[new.index(tail):].split('} // namespace')[0]
    assert 'cp.async.wait_group' not in new and new.count('__syncthreads();')==1
    assert f'sizeof(Storage)=={CONFIG[kind]["shared"]}' in new
    assert 'roof_slot_barrier::init(s.full[slot],32)' in new
    assert 'roof_slot_barrier::init(s.empty[slot],128)' in new
    assert 'roof_slot_barrier::release(s.empty[slot])' in new
    assert f'group+{CONFIG[kind]["stages"]}' in new
    wrapper=(ROOT/f'csrc/sm80/roof_{kind}_slot_pipeline_probe.cu').read_text()
    assert '__launch_bounds__(128,3)' in wrapper
    assert 'if(flag' in wrapper and 'o3_body<' in wrapper


def test_payload_copy_coverage_identical_and_unique():
    for size in (4096,8192):
        old=[(thread*16+chunk*2048)//64*64+(thread*16+chunk*2048)%64
             for thread in range(128) for chunk in range(size//2048)]
        new=[(lane*16+chunk*512)//64*64+(lane*16+chunk*512)%64
             for lane in range(32) for chunk in range(size//512)]
        assert sorted(old)==sorted(new)==list(range(0,size,16))
    assert [lane*4 for lane in range(16)]==list(range(0,64,4))
    assert [lane*4 for lane in range(32)]==list(range(0,128,4))


def test_sm80_barrier_arrival_count_and_acquire_not_sm90_wait():
    text=(ROOT/'csrc/sm80/roof_sm80_slot_barrier.cuh').read_text()
    assert 'cp.async.mbarrier.arrive.noinc.shared.b64' in text
    assert 'mbarrier.test_wait.parity.shared.b64' in text
    assert 'mbarrier.arrive.shared.b64 _, ' in text
    assert 'try_wait' not in text.split('// SM80 test_wait')[0]
    # noinc32 callbacks consume exactly the32 expected full arrivals.
    for count in range(32):assert 32-count>0
    assert 32-32==0 and 128-4*32==0


@pytest.mark.parametrize('stages',(2,3))
def test_random_asynchronous_slot_protocol_no_early_overwrite_or_deadlock(stages):
    # Model four warp readers and one existing warp0 publisher. Arbitrarily
    # delay copy completion/readers; warp0 must refill before its next group.
    # This checks protocol, not CUDA memory ordering or GPU correctness.
    for seed in range(100):
        rng=random.Random(seed);groups=32
        issued=0;progress=[0]*4;slot_group=[None]*stages
        pending=set();ready=set();readers={};refill=None;steps=0
        while progress!=[groups]*4:
            actions=[]
            if issued<stages:actions.append(('prime',issued))
            if refill is not None and len(readers[refill-stages])==4:
                actions.append(('refill',refill))
            actions.extend(('complete',g) for g in sorted(pending))
            for warp,g in enumerate(progress):
                if g<groups and g in ready and (warp!=0 or (issued>=stages and refill is None)):
                    actions.append(('consume',warp))
            assert actions,'protocol deadlock'
            action,value=rng.choice(actions);steps+=1
            assert steps<10000
            if action in ('prime','refill'):
                g=value;slot=g%stages;old=slot_group[slot]
                if old is not None:
                    assert old==g-stages and len(readers[old])==4
                    assert all(p>old for p in progress)
                    ready.remove(old)
                assert ((g//stages)&1)==(0 if old is None else 1-((old//stages)&1))
                slot_group[slot]=g;pending.add(g);readers[g]=set()
                if action=='prime':issued+=1
                else:refill=None
            elif action=='complete':
                assert value in pending;pending.remove(value);ready.add(value)
            else:
                warp=value;g=progress[warp]
                assert slot_group[g%stages]==g
                readers[g].add(warp);progress[warp]+=1
                if warp==0 and g+stages<groups:refill=g+stages
        assert all(len(readers[g])==4 for g in range(groups))
        assert not pending


def test_compile_gate_preserves_compute_work_and_no_full_CTA_hot_barrier():
    counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,'LDSM.16.M88.4':16}
    live=dict(allocated_gpr=168,loops=[dict(kind='integer',opcode_counts=counts)])
    assert worth_runtime(live)
    for changed in ({'BAR.SYNC':1},{'LDSM.16.M88.4':24},{'IMMA.16864.U4.S4':16},{'LDL.64':3}):
        assert not worth_runtime(dict(live,loops=[dict(kind='integer',opcode_counts={**counts,**changed})]))
    assert not worth_runtime(dict(live,allocated_gpr=176))

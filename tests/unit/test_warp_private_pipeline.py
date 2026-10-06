from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_warp_private_codegen import gate


def mock(registers=168,instructions=383,bar=0,warpsync=3,local=0):
    return dict(allocated_gpr=registers,loops=[dict(kind='integer',static_instructions=instructions,
        opcode_counts={'IMMA.16864.S4.S4':32,'IMMA.16864.U4.S4':32,
          'LDSM.16.M88.4':16,'BAR.SYNC':bar,'WARPSYNC':warpsync,'LDL':local})])


def test_gate_preserves_math_operand_work_resource_and_sync_budget():
    old=mock(bar=1,warpsync=0)
    assert gate(old,mock(instructions=430))['passed']
    for bad in (mock(registers=176),mock(instructions=441),mock(bar=1),mock(warpsync=0),mock(local=1)):
        assert not gate(old,bad)['passed']
    deferred=mock();deferred['loops'][0]['opcode_counts']['BAR.SYNC.DEFER_BLOCKING']=1
    assert not gate(old,deferred)['no_integer_CTA_barrier']


def test_private_payload_writes_do_not_touch_current_factor_storage():
    source=(ROOT/'csrc/sm80/o78_warp_private_pipeline.cuh').read_text()
    payload=source[source.index('__device__ __forceinline__ void payload'):source.index('__device__ __forceinline__ void factors')]
    assert 's.activation_factors' not in payload and 's.weight_factors' not in payload
    assert '__syncthreads' not in source and source.count('__syncwarp();')==3
    assert 'sizeof(Storage)==34304' in source
    assert 'partial(i)*=16' in source and 'group<32' in source
    assert 'acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient' in source
    assert 'partial[' not in source and '__launch_bounds__(128,3)' in (ROOT/'csrc/sm80/roof_o78_warp_private_probe.cu').read_text()


def test_each_private_warp_reads_all_bytes_once_and_duplicates_CTA_inputs_twice():
    seen_a={};seen_w={}
    for warp in range(4):
        row_base=(warp%2)*32;col_base=(warp//2)*64
        for lane in range(32):
            for chunk in range(4):
                off=lane*16+chunk*512
                for b in range(16):
                    key=(row_base+off//64,off%64+b)
                    seen_a[key]=seen_a.get(key,0)+1
            for chunk in range(8):
                off=lane*16+chunk*512
                for b in range(16):
                    key=(col_base+off//64,off%64+b)
                    seen_w[key]=seen_w.get(key,0)+1
    assert len(seen_a)==64*64 and set(seen_a.values())=={2}
    assert len(seen_w)==128*64 and set(seen_w.values())=={2}

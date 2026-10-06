from pathlib import Path
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from probe_operand_stream_codegen import CONTROL,SYMBOL,STEM,generated_header,worth_runtime
from probe_roof_fullk_integer_codegen import static_entries

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v94/reports'
DIR=EVIDENCE/'o378_roof_v94_codegen'


def test_exact_archived_sources_text_and_generated_header():
    r=json.loads((DIR/'codegen.json').read_text())
    assert r['source_commit']=='9a6d4a3ecf9502c1fd4a39952fbc7cf1e419f4d6'
    for name,digest in r['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in r['artifact_sha256'].items():
        if Path(name).suffix=='.cubin':continue  # Binary retained in the hashed complete archive.
        assert hashlib.sha256((DIR/name).read_bytes()).hexdigest()==digest,name
    assert (DIR/(STEM+'_generated.cuh')).read_text()==generated_header()
    assert not r['changed_semantics'] and not r['production_default_changed']


def test_same_native_MMA_but_gate_failed_from_real_machine_code():
    r=json.loads((DIR/'codegen.json').read_text())
    sass=(DIR/(STEM+'.sass')).read_text();ptx=(DIR/(STEM+'.ptx')).read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert entries==r['entries']
    for symbol in (CONTROL,SYMBOL):
        live=analyze((DIR/'liveness.txt').read_text(),symbol)
        assert live==r['liveness'][symbol]
        block=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        assert all(x in block for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
        assert entries[symbol]['native_u4_s4'] and entries[symbol]['native_s4_s4'] and not entries[symbol]['int8_mma']
    live=r['liveness'][SYMBOL];loop=next(x for x in live['loops'] if x['kind']=='integer')
    assert live['allocated_gpr']==168 and loop['max_live_gpr']==157
    assert loop['static_instructions']==380 and loop['opcode_counts']['LDSM.16.M88.4']==24
    assert not worth_runtime(live) and not r['worth_runtime_validation']
    assert not any(k.split('.')[0] in ('LDL','STL') for k in loop['opcode_counts'])
    prior=ROOT/'docs/evidence/a100_o378_roof_v93/reports/o378_roof_v93_o78_codegen/o78_stage_cycle.sass'
    assert compare(prior.read_text(),sass,'^'+CONTROL+'$')['passed']


def test_actual_scheduled_peak_is_eight_not_source_four():
    r=json.loads((DIR/'codegen.json').read_text());sass=(DIR/(STEM+'.sass')).read_text()
    for symbol,folder in ((SYMBOL,'o378_roof_v94_schedule'),(CONTROL,'o378_roof_v94_control_schedule')):
        stored=json.loads((EVIDENCE/folder/'analysis.json').read_text())
        actual=json.loads(json.dumps(trace(sass,symbol,r['liveness'][symbol])))
        for key,value in actual.items():assert stored[key]==value,key
        assert actual['total_mma']==64 and actual['chains_per_group']==16
        assert actual['peak_started_not_finished_chains']==8
        assert not stored['new_performance_result']
    assert r['partial_registers']==16 and r['independent_chains']==4


def test_driver_resource_query_and_entry_spill_not_runtime_bytes():
    query=json.loads((EVIDENCE/'o378_v94_occupancy_verified.json').read_text())
    assert query==dict(registers=168,local_bytes=8,threads=128,active_blocks_per_sm=3,
                       kind='CUDA_driver_resource_query_no_kernel_launch')
    build=(DIR/'build.log').read_text()
    start=build.index("Compiling entry function '"+SYMBOL+"'")
    block=build[start:build.index("Compiling entry function '"+CONTROL+"'",start)]
    assert '8 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads' in block
    r=json.loads((DIR/'codegen.json').read_text())
    fallback=next(x for x in r['liveness'][SYMBOL]['loops'] if x['kind']=='fp32_fallback')
    assert fallback['max_live_gpr']==160 and fallback['opcode_counts']['LDL']==2

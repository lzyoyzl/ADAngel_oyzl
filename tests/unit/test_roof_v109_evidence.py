"""Frozen compiler-only v109 evidence; no candidate GPU correctness claim."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_roof_fullk_integer_codegen import static_entries
from probe_warp_private_codegen import CONTROL,SYMBOL,STEM,gate

DATA=ROOT/'docs/evidence/a100_o378_roof_v109/reports/o378_roof_v109_codegen'


def receipt():return json.loads((DATA/'codegen.json').read_text())


def test_frozen_compiler_sources_artifacts_and_host_mapping():
    r=receipt()
    assert r['source_commit']=='8f7ac987db010819af05cfff22bb4c81ebc9dce8'
    for path,digest in r['sources'].items():
        # The sole subsequent change recognizes BAR.SYNC.DEFER_BLOCKING in
        # the CPU gate. Keep the original compilation receipt immutable.
        if path=='scripts/probe_warp_private_codegen.py':
            content=subprocess.check_output(['git','show',r['source_commit']+':'+path],cwd=ROOT)
        else:content=(ROOT/path).read_bytes()
        assert hashlib.sha256(content).hexdigest()==digest,path
    for name,digest in r['artifact_sha256'].items():
        if name=='validate_coordinates' or Path(name).suffix=='.cubin':continue
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    assert r['coordinates']==dict(passed=True,gpu_execution=False,output_elements=8192,
        old_and_private_unique_owners=True,ownership_changed=True,copy_byte_checks=6144,shared_bytes=34304)
    assert r['cta_tile']==[64,128,128] and r['threads']==128 and r['private_buffer_depth']==1
    assert r['payload_global_copy_bytes_ratio']==2 and r['partial_registers']==32
    assert r['independent_chains']==8 and r['G128_order_unchanged']
    assert not r['production_default_changed']


def test_exact_entry_INT4_and_preserved_best_control_words():
    r=receipt();sass=(DATA/(STEM+'.sass')).read_text();ptx=(DATA/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert es==r['entries']
    for symbol in (CONTROL,SYMBOL):
        e=es[symbol]
        assert e['native_s4_s4'] and e['native_u4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
            if b.startswith('.visible .entry '+symbol+'('))
        assert all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
        if symbol==SYMBOL:assert body.count('bar.warp.sync')==3
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']


def test_compiler_work_and_hot_local_fail_without_GPU_claim():
    r=receipt();live=(DATA/'liveness.txt').read_text()
    for symbol in (CONTROL,SYMBOL):assert analyze(live,symbol)==r['liveness'][symbol]
    old,new=(r['liveness'][s] for s in (CONTROL,SYMBOL))
    assert old['allocated_gpr']==new['allocated_gpr']==168
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    assert (a['static_instructions'],b['static_instructions'])==(383,433)
    assert (a['max_live_gpr'],b['max_live_gpr'])==(166,164)
    assert a['opcode_counts']['LDGSTS.E.BYPASS.128']==10
    assert b['opcode_counts']['LDGSTS.E.BYPASS.128']==18
    assert sum(v for k,v in b['opcode_counts'].items() if k.startswith(('LDL','STL')))==8
    assert not any(v for k,v in b['opcode_counts'].items() if k.startswith('BAR.SYNC'))
    assert b['opcode_counts'].get('WARPSYNC',0)==0
    assert gate(old,new)==r['compile_gate'] and not gate(old,new)['passed']
    build=(DATA/'build.log').read_text();lo=build.index("Compiling entry function '"+SYMBOL+"'")
    hi=build.index("Compiling entry function '"+CONTROL+"'",lo)
    assert '32 bytes stack frame, 44 bytes spill stores, 36 bytes spill loads' in build[lo:hi]


def test_one_completed_build_not_a_runtime_performance_test():
    log=(DATA.parent/'o378_roof_v109_codegen.log').read_text()
    assert '"compile_gate"' in log and 'Traceback' not in log
    assert 'compute_only' not in log and 'median_ms' not in log
    assert len(receipt()['commands'])==7

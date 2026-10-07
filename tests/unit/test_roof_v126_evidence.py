"""Replay v126 conversion codegen and stop gate; no GPU/performance claims."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v126'
B=E/'reports/o378_roof_v126_codegen_r2'
FIRST=E/'reports/o378_roof_v126_codegen'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_texts_and_compiled_source_identity():
    index=read(E/'index.json');r=read(B/'build.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert index['artifact_count']==len(index['files'])==19
    assert sum(f['bytes'] for f in index['files'])==4692686
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    assert r['source_commit']==index['compile_source_commit']=='436e974e355d3c3e85eea32603db925546ff8109'
    for f,digest in r['sources'].items():
        raw=subprocess.check_output(['git','show',r['source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for f,digest in r['artifact_sha256'].items():
        if not f.endswith('.so'):assert sha(B/f)==digest
    assert r['driver_sha256']=='906dde016ff9516edda4f1ee682932dd5816f0706c66c698ea1e0c846cc837fe'
    assert not any(index[k] for k in ('candidate_GPU_executed','new_performance_or_MSE_results',
                                   'production_default_changed','candidate_adopted'))
    assert not r['GEMM_modified'] and not r['production_default_changed']


def test_real_entry_work_resources_controls_and_failed_gate():
    from probe_mx8_swar_codegen import find_mx8_entry
    from analyze_o78_row_fused_codegen import entries
    from compare_a100_codegen import compare
    r=read(B/'build.json');a=r['audit'];text=(B/'prepare.sass').read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v106/reports/o378_roof_v106_codegen/prepare.sass'
    wc=ROOT/'docs/evidence/a100_o378_roof_v118/reports/o378_roof_v118_codegen/prepare.sass'
    control,ce=find_mx8_entry(old.read_text(),'adangel_sm80_row_warp_lut_metadata')
    candidate,ne=find_mx8_entry(text,'adangel_sm80_mx8_swar_metadata')
    assert ce=={k:a['control'][k] for k in ce} and ne=={k:a['candidate'][k] for k in ne}
    assert compare(old.read_text(),text,'^'+re.escape(control)+'$')==a['old_controls']
    weight=next(s for s in entries(wc.read_text()) if 'adangel_sm80_row_swar_metadata' in s and 'GroupedSourceKindE0E' in s)
    assert compare(wc.read_text(),text,'^'+re.escape(weight)+'$')==a['weight_control']
    assert a['old_controls']['passed'] and a['weight_control']['passed']
    assert (ce['instructions'],ne['instructions'])==(456,464)
    assert a['static_instruction_reduction_fraction']==1-464/456
    resources={s:dict(registers=int(r),stack=int(st),shared=int(sh),local=int(lo)) for s,r,st,sh,lo
        in re.findall(r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',(B/'resources.txt').read_text())}
    for label,symbol in (('control',control),('candidate',candidate)):
        expected=dict(registers=31,stack=0,shared=256,local=0)
        assert resources[symbol]==expected
        assert {k:a[label][k] for k in expected}==expected
        runtime=a['runtime_resources'][label]
        assert {k:runtime[k] for k in ('registers','shared','local')}==dict(registers=31,shared=256,local=0)
        assert runtime['active_blocks_per_sm']==8
    ops=ne['opcode_counts']
    assert ops['IDP.4A.S8.S8']==4 and ops['SHFL.IDX']==4 and ops['LOP3.LUT']==122
    assert not any(op.startswith(('LDL','STL')) for op in ops)
    assert sum(v for op,v in ops.items() if op.startswith('BAR.SYNC'))==1
    assert not a['gate']['work_reduction_at_least5pct'] and not a['worth_runtime_validation']
    assert all(v for k,v in a['gate'].items() if k!='work_reduction_at_least5pct')
    assert a['no_candidate_kernel_launched']


def test_failed_first_audit_not_hidden_and_cuda_algorithm_unchanged():
    from probe_mx8_swar_codegen import generated_files,find_mx8_entry
    from compare_a100_codegen import compare
    log=(E/'tmp/o378_v126_codegen.log').read_text()
    assert 'ValueError: one MX8 entry required' in log and not (FIRST/'build.json').exists()
    for name,value in generated_files().items():
        assert (FIRST/name).read_text()==(B/name).read_text()==value
    symbol,_=find_mx8_entry((B/'prepare.sass').read_text(),'adangel_sm80_mx8_swar_metadata')
    assert compare((FIRST/'prepare.sass').read_text(),(B/'prepare.sass').read_text(),'^'+re.escape(symbol)+'$')['passed']

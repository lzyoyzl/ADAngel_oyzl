"""CPU-only reproduction of frozen O3 transpose codegen, not GPU validation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v124'
B=E/'reports/o378_roof_v124_codegen'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_texts_source_and_binary_references():
    index=read(E/'index.json');receipt=read(B/'codegen.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert len(index['files'])==index['artifact_count']==16
    assert receipt['source_commit']==index['compile_source_commit']
    for f in index['files']:
        path=E/f['path']
        assert path.relative_to(ROOT).as_posix() in tracked
        assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
    for f,digest in receipt['sources'].items():
        raw=subprocess.check_output(['git','show',index['compile_source_commit']+':'+f],cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest()==digest
    for f,digest in receipt['artifact_sha256'].items():
        if f not in ('validate_coordinates','o3_transposed_mma.cubin'):assert sha(B/f)==digest
    assert not index['candidate_GPU_executed'] and not index['new_performance_or_MSE_results']
    assert not index['production_default_changed']


def test_recompute_liveness_gate_and_control():
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_transposed_mma_codegen import CONTROL,SYMBOL,cost_gate
    r=read(B/'codegen.json');text=(B/'liveness.txt').read_text()
    live={s:analyze(text,s) for s in (CONTROL,SYMBOL)}
    assert live==r['liveness']
    old=ROOT/'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    comparison=compare(old.read_text(),(B/'o3_transposed_mma.sass').read_text(),'^'+CONTROL+'$')
    assert comparison==r['control_comparison'] and comparison['passed']
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=True
    assert gate==r['cost_gate'] and not gate['passed']
    assert not gate['checks']['meaningful_work_or_live_reduction']
    assert all(v for k,v in gate['checks'].items() if k!='meaningful_work_or_live_reduction')
    a,b=(next(x for x in live[s]['loops'] if x['kind']=='integer') for s in (CONTROL,SYMBOL))
    assert (a['static_instructions'],b['static_instructions'])==(323,322)
    assert (a['max_live_gpr'],b['max_live_gpr'])==(166,160)
    assert a['opcode_counts']['LDS.64']==8 and b['opcode_counts']['LDS']==11
    assert a['opcode_counts']['IMAD']==b['opcode_counts']['IMAD']==65


def test_cute_host_coordinates_native_S4_U4_and_unmodified_generated_controls():
    from probe_transposed_mma_codegen import CONTROL,SYMBOL
    from probe_grouped_cta_codegen import generated_headers
    from probe_roof_fullk_integer_codegen import static_entries
    r=read(B/'codegen.json');coords=read(B/'coordinates.log')
    assert coords==r['coordinates'] and coords['passed'] and not coords['gpu_execution']
    assert (coords['outputs'],coords['operand_coordinates_compared'],coords['weight_scales_per_thread'])==(8192,49152,8)
    sass=(B/'o3_transposed_mma.sass').read_text();ptx=(B/'o3_transposed_mma.ptx').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    candidate=next(b for b in re.split(r'(?=Function\s*:\s*)',sass)
        if b.startswith('Function') and b.splitlines()[0].split(':',1)[1].strip()==SYMBOL)
    entries[SYMBOL]['native_s4_u4']=bool(re.search(r'\bIMMA[^;]*\.S4\.U4',candidate))
    assert entries==r['entries'] and entries[SYMBOL]['native_s4_u4'] and not entries[SYMBOL]['int8_mma']
    entry=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.s4.u4.s32','.s32.s4.s4.s32'))
    for name,body in zip(('o3_grouped_cta_generated.cuh','o3_grouped_fallback_generated.cuh'),generated_headers('o3')):
        assert (B/name).read_text()==body

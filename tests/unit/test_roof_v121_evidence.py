"""Frozen one-off supply gate; no CUDA launch or new latency assertion."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_o78_readonly_factor_codegen as probe
from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions,trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header

E=ROOT/'docs/evidence/a100_o378_roof_v121'
D=E/'reports/o378_roof_v121_codegen'
R=json.loads((D/'codegen.json').read_text())
INDEX=json.loads((E/'index.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()


def test_frozen_raw_files_pinned_source_and_no_runtime_claim():
    assert INDEX['source_commit']==R['source_commit']=='2a36f6aec73bf9329d86caec854d141615decb1e'
    assert not INDEX['new_performance_or_MSE'] and not INDEX['production_default_changed']
    assert len(INDEX['files'])==13
    tracked={p.decode() for p in subprocess.check_output(['git','ls-files','-z','--',
        E.relative_to(ROOT).as_posix()],cwd=ROOT).split(b'\0') if p}
    for f in INDEX['files']:
        assert (E/f['path']).relative_to(ROOT).as_posix() in tracked,f['path']
        data=(E/f['path']).read_bytes()
        assert sha(data)==f['sha256'] and len(data)==f['bytes']
    for name,value in R['sources'].items():
        assert sha(subprocess.check_output(['git','show',R['source_commit']+':'+name],cwd=ROOT))==value,name
    for name,value in R['artifact_sha256'].items():
        assert sha((D/name).read_bytes())==value,name
    assert sha((D/'o78_readonly_factor_generated.cuh').read_bytes())==R['generated_header_sha256']
    assert not (D/'o78_readonly_factor.cubin').exists()
    assert (D/'server_head.txt').read_text().strip()==R['source_commit']
    assert (D/'cutlass_commit.txt').read_text().strip()==R['cutlass_commit']
    assert (D/'formal_extension_sha256.txt').read_text().split()[0]==(
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')


def test_exact_generated_body_and_encoded_v78_control():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    assert generated_header(source)==(D/'o78_eight_chain_generated.cuh').read_text()
    assert probe.generated_header(source)==(D/'o78_readonly_factor_generated.cuh').read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),(D/'o78_readonly_factor.sass').read_text(),
        '^'+probe.CONTROL+'$')==R['control_comparison']
    assert R['control_comparison']['passed']


def test_native_math_and_factor_supply_cost_failure_replay():
    sass=(D/'o78_readonly_factor.sass').read_text();text=(D/'liveness.txt').read_text()
    live={s:analyze(text,s) for s in (probe.CONTROL,probe.SYMBOL)}
    assert live==R['liveness']
    assert all(v['allocated_gpr']==168 for v in live.values())
    schedules={s:trace(sass,s,live[s]) for s in live}
    assert json.loads(json.dumps(schedules))==R['schedules']
    assert all(v['peak_started_not_finished_chains']==8 for v in schedules.values())
    gate=probe.cost_gate(live[probe.CONTROL],live[probe.SYMBOL]);gate['control_encoding_unchanged']=True
    assert gate==R['cost_gate'] and not gate['passed']
    assert gate['old_static']==383 and gate['new_static']==534
    assert not gate['checks']['meaningful_work_reduction']
    assert all(v for k,v in gate['checks'].items() if k!='meaningful_work_reduction')
    loop=next(l for l in live[probe.SYMBOL]['loops'] if l['kind']=='integer')
    scalar_shared=[t for _,t in instructions(sass,probe.SYMBOL,loop) if t.split()[0].startswith('LDS')
        and not t.split()[0].startswith('LDSM')]
    assert scalar_shared==['LDS RZ, [RZ]']*3
    assert loop['opcode_counts']['LDG.E.CONSTANT']==20
    assert 'ld.global.nc' in (D/'o78_readonly_factor.ptx').read_text()
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and
        e['all_copies_bypass_l1'] for e in R['entries'].values())

"""Replay failed compile gate without launching a candidate or claiming gain."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_o78_phase_mixed_codegen as probe
from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header

E=ROOT/'docs/evidence/a100_o378_roof_v120'
D=E/'reports/o378_roof_v120_codegen'
R=json.loads((D/'codegen.json').read_text())
INDEX=json.loads((E/'index.json').read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()


def test_frozen_raw_files_source_commit_and_not_runtime_scope():
    assert INDEX['source_commit']==R['source_commit']=='9d9b8a103caff66318da3e2f63c3023c92bb97f8'
    assert not INDEX['new_performance_or_MSE'] and not INDEX['production_default_changed']
    assert len(INDEX['files'])==14
    tracked={p.decode() for p in subprocess.check_output(['git','ls-files','-z','--',
        E.relative_to(ROOT).as_posix()],cwd=ROOT).split(b'\0') if p}
    for f in INDEX['files']:
        assert (E/f['path']).relative_to(ROOT).as_posix() in tracked,f['path']
        data=(E/f['path']).read_bytes()
        assert sha(data)==f['sha256'] and len(data)==f['bytes']
    for name,value in R['sources'].items():
        data=subprocess.check_output(['git','show',R['source_commit']+':'+name],cwd=ROOT)
        assert sha(data)==value,name
    for name,value in (R['artifact_sha256']|R['generated_sha256']).items():
        assert sha((D/name).read_bytes())==value,name
    assert not (D/'o78_phase_mixed.cubin').exists()
    assert (D/'server_head.txt').read_text().strip()==R['source_commit']
    assert (D/'cutlass_commit.txt').read_text().strip()==R['cutlass_commit']
    assert (D/'formal_extension_sha256.txt').read_text().split()[0]==(
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')


def test_exact_generated_headers_and_original_control_encoding_unchanged():
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    assert generated_header(source)==(D/'o78_eight_chain_generated.cuh').read_text()
    assert probe.generated_four(source)==(D/'o78_four_chain_generated.cuh').read_text()
    baseline=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(baseline.read_text(),(D/'o78_phase_mixed.sass').read_text(),
        '^'+probe.CONTROL+'$')==R['control_comparison']
    assert R['control_comparison']['passed']


def test_replay_actual_paths_hot_local_and_failed_fixed_gate():
    sass=(D/'o78_phase_mixed.sass').read_text();text=(D/'liveness.txt').read_text()
    old=analyze(text,probe.CONTROL);new=probe.mixed_liveness(text)
    assert old==R['liveness'][probe.CONTROL] and new==R['liveness'][probe.SYMBOL]
    assert old['allocated_gpr']==new['allocated_gpr']==168
    paths=probe.schedules(sass,new)
    assert json.loads(json.dumps(paths))==R['schedules']
    assert [p['peak_started_not_finished_chains'] for p in paths]==[8,6]
    assert [p['loop']['static_instructions'] for p in paths]==[368,367]
    assert [p['loop']['opcode_counts']['LDL.64'] for p in paths]==[4,3]
    gate=probe.cost_gate(old,new,paths);gate['control_encoding_unchanged']=True
    assert gate==R['cost_gate'] and not gate['passed']
    assert gate['checks']['same_math'] and gate['checks']['same_supply']
    assert gate['checks']['same_barrier'] and gate['checks']['allocation']
    assert not gate['checks']['no_hot_local'] and not gate['checks']['distinct_four_and_eight_paths']
    assert gate['weighted_static'] is None and gate['weighted_work_ratio'] is None
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and
        e['all_copies_bypass_l1'] for e in R['entries'].values())

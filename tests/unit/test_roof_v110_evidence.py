"""Frozen v110 capacity evidence, not real-trace speedup or MSE evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from probe_o78_shell_capacity import CONTROL,SYMBOLS,STEM,compile_gate,hot_loop
from probe_roof_fullk_integer_codegen import static_entries
from run_o78_shell_capacity import summarize

DATA=ROOT/'docs/evidence/a100_o378_roof_v110/reports'
COMPILES=('o378_roof_v110_shell_capacity','o378_roof_v110_shell_capacity_r2',
          'o378_roof_v110_shell_capacity_r3')
COMMITS=('630dfe4e704e326bae292291fa2859e29f1962a2',
         '9ce9cdd1c7379e69cc7e224e8be3c67d47fb0ed2',
         'e86e28b3a63b06357b423239cbd5e82ee77a6f94')


def receipt(name):return json.loads((DATA/name/'analysis.json').read_text())


def verify_hashes(directory,record,binary_names):
    for path,digest in record['sources'].items():
        original=subprocess.check_output(['git','show',record['source_commit']+':'+path],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,path
    for name,digest in record['artifact_sha256'].items():
        if name in binary_names:continue  # Binary archive is SHA-identified, not in Git.
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name


def test_all_three_compile_receipts_and_repair_logs_are_preserved():
    for name,commit in zip(COMPILES,COMMITS):
        r=receipt(name)
        assert r['source_commit']==commit
        verify_hashes(DATA/name,r,{STEM+'.cubin'})
        assert not r['new_MSE_measured'] and not r['new_real_GEMM_measured']
        assert not r['production_default_changed']
        assert r['extension_sha256_before']==r['extension_sha256_after']
        assert 'runtime' not in r  # Whole three-shell compile gate failed; no old launch.
    assert 'warning #836-D' in (DATA/COMPILES[0]/'build.log').read_text()
    for name in COMPILES[1:]:
        assert 'warning #836-D' not in (DATA/name/'build.log').read_text()


def test_same_entry_native_INT4_and_best_control_words():
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    for name in COMPILES:
        r=receipt(name);sass=(DATA/name/(STEM+'.sass')).read_text()
        assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
        assert r['control_comparison']['passed']
        entries=static_entries(sass,'^(?:'+'|'.join(SYMBOLS)+')$',set(SYMBOLS))
        assert entries==r['entries']
        for entry in entries.values():
            assert entry['native_s4_s4'] and entry['native_u4_s4'] and not entry['int8_mma']


def test_gate_not_relaxed_and_scaled_path_stopped():
    for name in COMPILES:
        r=receipt(name);live=(DATA/name/'liveness.txt').read_text()
        rows={s:hot_loop(live,s) for s in SYMBOLS}
        assert rows==r['liveness'] and compile_gate(rows)==r['compile_gate']
        assert not r['compile_gate']['passed']
        for check in r['compile_gate']['checks'][:2]:
            assert all(v for k,v in check.items() if k!='symbol')
        assert not r['compile_gate']['checks'][2]['no_hot_local']
    r=receipt(COMPILES[2]);loops=[r['liveness'][s] for s in SYMBOLS]
    assert [x['allocated_gpr'] for x in loops]==[166,165,168]
    assert [x['loop']['static_instructions'] for x in loops]==[195,213,300]
    assert [x['loop']['max_live_gpr'] for x in loops]==[164,151,164]
    for mode,x in enumerate(loops):
        ops=x['loop']['opcode_counts']
        assert ops['IMMA.16864.S4.S4']==ops['IMMA.16864.U4.S4']==32
        assert sum(v for k,v in ops.items() if k.startswith('LDSM'))==(0 if mode==0 else 16)
        assert sum(v for k,v in ops.items() if k.startswith(('LDL','STL')))==(6 if mode==2 else 0)


def test_two_shells_same_cubin_host_only_rebuild_full_raw_events():
    name='o378_roof_v110_two_shells';r=receipt(name);prior=receipt(COMPILES[2])
    verify_hashes(DATA/name,r,{'capacity_driver'})
    assert r['source_commit']=='637434204bd7081c2f14512f0128ce94a478b4a7'
    assert r['compile_receipt_sha256']==hashlib.sha256((DATA/COMPILES[2]/'analysis.json').read_bytes()).hexdigest()
    assert r['cubin_sha256']==prior['artifact_sha256'][STEM+'.cubin']
    assert r['selected_modes']==[0,1] and r['rejected_scaled_shell_not_launched']
    assert r['static_gate_not_relaxed'] and not r['production_default_changed']
    assert r['commands'][0][3].endswith('roof_o78_shell_capacity_driver.cpp')
    assert all(not arg.endswith('.cu') for cmd in r['commands'] for arg in cmd)
    assert r['commands'][2][-1]=='3'
    assert r['extension_sha256_before']==r['extension_sha256_after']==prior['extension_sha256_after']
    rows=[json.loads(x) for x in (DATA/name/'results.jsonl').read_text().splitlines()]
    assert summarize(rows)==r['runtime'] and len(rows)==6
    assert [x['registers'] for x in rows]==[166,165]*3
    assert all(x['local_bytes']==0 and x['active_ctas_per_sm']==3 and x['warmup']==50 for x in rows)
    assert all(x['validation_checks']==12 and x['checksum_passed'] for x in rows)
    assert sum(len(x['raw_ms']) for x in rows)==1200
    assert max(x['cv_percent'] for x in r['runtime']['statistics'])<3
    assert r['runtime']['modes'][0]['normalized32_groups_ms']==0.29311999625
    assert r['runtime']['modes'][1]['normalized32_groups_ms']==0.295551985
    assert not r['runtime']['original_experiment_MSE_measured']
    assert not r['runtime']['real_trace_latency_measured']
    assert r['runtime']['no_filtering'] and r['runtime']['normalized_times_are_diagnostic_not_kernel_peak']

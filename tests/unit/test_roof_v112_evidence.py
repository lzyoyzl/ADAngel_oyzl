"""Frozen v112 compile rejection, never GPU numerical/performance acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from probe_cooperative_reuse_codegen import analyze
from probe_direct_fragment_codegen import CONTROL,SYMBOL,STEM,compile_gate,cost_model,header
from probe_roof_fullk_integer_codegen import static_entries

DATA=ROOT/'docs/evidence/a100_o378_roof_v112/reports/o378_roof_v112_direct_fragment_codegen'


def receipt():return json.loads((DATA/'codegen.json').read_text())


def test_frozen_source_and_readable_artifact_hashes():
    r=receipt()
    assert r['source_commit']=='d3537a05430a7efc8a823b42667eccd4550cdbcb'
    for name,digest in r['sources'].items():
        original=subprocess.check_output(['git','show',r['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    for name,digest in r['artifact_sha256'].items():
        if name in {STEM+'.cubin','verify_coordinates'}:continue
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    old=ROOT/'docs/evidence/a100_o378_roof_v94/reports/o378_roof_v94_codegen/o78_operand_stream_generated.cuh'
    assert header(old.read_text())==(DATA/(STEM+'_generated.cuh')).read_text()


def test_same_entry_native_INT4_and_control_machine_words():
    r=receipt();sass=(DATA/(STEM+'.sass')).read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert entries==r['entries']
    for e in entries.values():
        assert e['native_s4_s4'] and e['native_u4_s4'] and not e['int8_mma']


def test_failed_gate_not_relaxed_for_removed_shared_payload():
    r=receipt();live=(DATA/'liveness.txt').read_text()
    rows={s:analyze(live,s) for s in (CONTROL,SYMBOL)}
    assert rows==r['liveness']
    assert compile_gate(rows[CONTROL],rows[SYMBOL])==r['compile_gate']
    assert [rows[s]['allocated_gpr'] for s in (CONTROL,SYMBOL)]==[168,128]
    assert [rows[s]['loop']['static_instructions'] for s in (CONTROL,SYMBOL)]==[383,440]
    assert [rows[s]['loop']['max_live_gpr'] for s in (CONTROL,SYMBOL)]==[166,126]
    ops=rows[SYMBOL]['loop']['opcode_counts']
    assert sum(v for k,v in ops.items() if k.startswith('LDL'))==30
    assert sum(v for k,v in ops.items() if k.startswith('STL'))==22
    checks=r['compile_gate']['checks']
    assert checks['four_CTA_register_budget'] and checks['native_INT4_64']
    assert checks['no_hot_payload_shared_path']
    assert not checks['no_hot_local'] and not checks['fewer_instructions']
    assert not r['compile_gate']['passed']
    assert r['work_model']==cost_model()


def test_mapping_only_no_runtime_MSE_or_default_change():
    r=receipt();mapping=json.loads((DATA/'mapping.json').read_text())
    assert mapping==r['mapping'] and mapping['passed'] and not mapping['gpu_execution']
    assert mapping['a_register_nibbles']==16384 and mapping['b_register_nibbles']==32768
    assert mapping['legacy_adjacent_b_mismatches']==24576
    for name in ('candidate_gpu_launched','new_GEMM_measured','new_MSE_measured',
                 'production_default_changed'):
        assert not r[name]
    assert r['extension_sha256_before']==r['extension_sha256_after']==(
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
    assert r['shared_bytes']==24576
    assert 'runtime' not in r and 'results.jsonl' not in r['artifact_sha256']

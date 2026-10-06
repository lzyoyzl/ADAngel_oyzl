"""Frozen v111 compiler rejection; no GPU performance or numerical acceptance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from probe_cooperative_reuse_codegen import CONTROL,SYMBOL,STEM,analyze,gate,generated_header,work_model
from probe_roof_fullk_integer_codegen import static_entries

DATA=ROOT/'docs/evidence/a100_o378_roof_v111/reports/o378_roof_v111_cooperative_reuse_codegen'


def receipt():return json.loads((DATA/'codegen.json').read_text())


def test_frozen_source_and_all_readable_artifact_hashes():
    r=receipt()
    assert r['source_commit']=='b79da2ae1aa660c6d392af95262149c7dad0d5c4'
    for name,digest in r['sources'].items():
        original=subprocess.check_output(['git','show',r['source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(original).hexdigest()==digest,name
    for name,digest in r['artifact_sha256'].items():
        if name in {STEM+'.cubin','validate_coordinates'}:continue
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest,name
    old=(DATA/'o78_eight_chain_generated.cuh').read_text()
    assert generated_header(old)==(DATA/(STEM+'_generated.cuh')).read_text()


def test_same_entry_native_INT4_and_original_control_machine_words():
    r=receipt();sass=(DATA/(STEM+'.sass')).read_text()
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert entries==r['entries']
    for e in entries.values():
        assert e['native_s4_s4'] and e['native_u4_s4'] and not e['int8_mma']
        assert e['all_copies_bypass_l1']


def test_gate_not_relaxed_for_logical_memory_read_reduction():
    r=receipt();live=(DATA/'liveness.txt').read_text()
    rows={s:analyze(live,s) for s in (CONTROL,SYMBOL)}
    assert rows==r['liveness']
    assert gate(rows[CONTROL],rows[SYMBOL])==r['compile_gate']
    assert [rows[s]['allocated_gpr'] for s in (CONTROL,SYMBOL)]==[168,168]
    assert [rows[s]['loop']['static_instructions'] for s in (CONTROL,SYMBOL)]==[383,387]
    assert [rows[s]['loop']['max_live_gpr'] for s in (CONTROL,SYMBOL)]==[166,162]
    checks=r['compile_gate']['checks']
    assert not checks['weighted_instructions_at_least5pct_less']
    assert all(v for k,v in checks.items() if k!='weighted_instructions_at_least5pct_less')
    assert not r['compile_gate']['passed']
    assert r['work_model']==work_model()
    assert r['work_model']['input_bytes_ratio']<.60
    assert r['compile_gate']['weighted_static_instruction_ratio']==1.0420202349869452


def test_host_mapping_only_no_candidate_runtime_MSE_or_default_change():
    r=receipt();coords=json.loads((DATA/'coordinates.log').read_text())
    assert coords==r['coordinates'] and coords['passed'] and not coords['gpu_execution']
    assert coords['output_elements']==24576 and coords['vector_pairs']==12288
    assert coords['copy_byte_owners']==20480 and coords['nibble_layout_checks']==40960
    assert r['shared_bytes']==coords['shared_bytes']==59904
    assert r['full_fallback_integration_pending']
    for name in ('candidate_gpu_launched','new_real_GEMM_measured','new_MSE_measured',
                 'production_default_changed','native_extension_rebuilt','per_group_scale_semantics_changed'):
        assert not r[name]
    assert r['extension_sha256_before']==r['extension_sha256_after']==(
        '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462')
    assert 'runtime' not in r and 'results.jsonl' not in r['artifact_sha256']

"""Replay frozen v129 compile result; no candidate GPU test was performed."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v129'
B = E / 'reports/o378_roof_v129_codegen'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_texts_sources_and_non_gpu_scope():
    index = read(E / 'index.json')
    r = read(B / 'codegen.json')
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '-z', '--', str(E.relative_to(ROOT))],
        cwd=ROOT).decode().split('\0'))
    assert index['artifact_count'] == len(index['files']) == 14
    assert index['artifact_bytes'] == sum(f['bytes'] for f in index['files']) == 7105468
    for f in index['files']:
        p = E / f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256']
    assert r['source_commit'] == index['compile_source_commit'] == '182f79b29c3662475b1cfa1aab88ec8fbb191e79'
    for f, digest in r['sources'].items():
        raw = subprocess.check_output(['git', 'show', r['source_commit'] + ':' + f], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == digest
    for f, digest in r['artifact_sha256'].items():
        if f != 'coordinate_check' and not f.endswith('.cubin'):
            assert sha(B / f) == digest
    assert not any(index[k] for k in ('candidate_GPU_executed', 'new_performance_or_MSE_results',
        'new_NCU_result', 'production_default_changed', 'candidate_adopted'))
    assert not any(r[k] for k in ('new_candidate_GPU_executed', 'production_default_changed',
        'conversion_changed', 'changed_semantics'))
    assert index['formal_extension_sha256'] == '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert r['coordinates']['passed'] and not r['coordinates']['gpu_execution']
    assert r['coordinates']['outputs'] == 8192 and r['coordinates']['C_matches_payload']


def test_real_helper_math_cost_and_failed_gate():
    from probe_o78_tensor_factor_codegen import CONTROL, SYMBOL, STEM, generated_header, analyze_candidate, cost_gate
    from inspect_o78_register_liveness import analyze
    from compare_a100_codegen import compare
    from probe_roof_fullk_integer_codegen import static_entries
    r = read(B / 'codegen.json')
    sass = (B / (STEM + '.sass')).read_text()
    old = ROOT / 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(), sass, '^' + CONTROL + '$') == r['control_comparison']
    assert r['control_comparison']['passed']
    entries = static_entries(sass, '^(?:' + CONTROL + '|' + SYMBOL + ')$', {CONTROL, SYMBOL})
    assert entries == r['entries']
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and e['all_copies_bypass_l1'] for e in entries.values())
    assert entries[SYMBOL]['int8_mma'] and not entries[CONTROL]['int8_mma']
    live_text = (B / 'liveness.txt').read_text()
    live = {CONTROL: analyze(live_text, CONTROL), SYMBOL: analyze_candidate(live_text)}
    assert live == r['liveness']
    gate = cost_gate(live[CONTROL], live[SYMBOL])
    gate['control_encoding_unchanged'] = True
    assert gate == r['cost_gate'] and not gate['passed']
    assert {k for k, v in gate['checks'].items() if not v} == {'work', 'hot_local'}
    assert (gate['old_static'], gate['new_static'], gate['old_imad'], gate['new_imad']) == (383, 461, 158, 115)
    assert live[CONTROL]['allocated_gpr'] == live[SYMBOL]['allocated_gpr'] == 168
    loop = next(x for x in live[SYMBOL]['loops'] if x['kind'] == 'tensor_factor_integer')
    c = loop['opcode_counts']
    assert loop['max_live_gpr'] == 166
    assert (c['PRMT'], c['LDL.LU'], c['STL'], c['IMMA.16816.U8.U8']) == (54, 3, 3, 16)
    assert c['IMMA.16864.S4.S4'] == c['IMMA.16864.U4.S4'] == 32
    assert (B / (STEM + '_generated.cuh')).read_text() == generated_header()
    entry = next(x for x in re.split(r'(?=\.visible \.entry )', (B / (STEM + '.ptx')).read_text())
        if x.startswith('.visible .entry ' + SYMBOL + '('))
    assert all(x in entry for x in ('cp.async.cg.shared.global', '.s32.u4.s4.s32',
        '.s32.s4.s4.s32', 'mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32'))
    assert '32 bytes stack frame, 76 bytes spill stores, 64 bytes spill loads' in (B / 'build.log').read_text()
    assert 'FAILED: stop without candidate GPU execution or neighboring layout scan' in (E / 'tmp/o378_v129_codegen.log').read_text()

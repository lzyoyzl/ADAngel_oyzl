"""Replay v140 source, native instructions and failed predeclared gate."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
E = ROOT / 'docs/evidence/a100_o378_roof_v140'
B = E / 'reports/o378_roof_v140_codegen'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())


def test_raw_evidence_matches_compiled_commit_and_has_no_performance_claim():
    index, r = read(E / 'index.json'), read(B / 'codegen.json')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z', '--', str(E.relative_to(ROOT))],
                                         cwd=ROOT).decode().split('\0'))
    assert index['artifact_count'] == len(index['files'])
    assert index['artifact_bytes'] == sum(f['bytes'] for f in index['files'])
    for f in index['files']:
        p = E / f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256']
    assert r['source_commit'] == index['compile_source_commit'] == 'a213dc08e0acc61427fc4449dc7555ee32d57fca'
    for name, digest in r['sources'].items():
        raw = subprocess.check_output(['git', 'show', r['source_commit'] + ':' + name], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == digest
    for name, digest in r['artifact_sha256'].items():
        if name.endswith(('.json', '.log', '.cu', '.cuh', '.txt', '.sass', '.ptx')):
            assert sha(B / name) == digest
    assert not any(index[k] for k in ('candidate_GPU_executed', 'new_performance_or_MSE_results',
        'new_NCU_result', 'production_default_changed', 'candidate_adopted'))


def test_same_math_no_modulo_but_more_work_and_hot_spill():
    from compare_a100_codegen import compare
    from inspect_eight_chain_schedule import instructions, trace
    from inspect_o78_register_liveness import analyze
    from probe_o3_ring_counter_codegen import CONTROL, SYMBOL, STEM, generated_header, cost_gate
    from probe_roof_fullk_integer_codegen import static_entries
    r = read(B / 'codegen.json')
    sass = (B / (STEM + '.sass')).read_text()
    prior = ROOT / 'docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
    assert compare(prior.read_text(), sass, '^' + CONTROL + '$') == r['control_comparison']
    assert r['control_comparison']['passed']
    es = static_entries(sass, '^(?:' + CONTROL + '|' + SYMBOL + ')$', {CONTROL, SYMBOL})
    assert es == r['entries'] and all(v['native_u4_s4'] and v['native_s4_s4'] and
        not v['int8_mma'] and v['all_copies_bypass_l1'] for v in es.values())
    live = {s: analyze((B / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)}
    assert live == r['liveness']
    gate = cost_gate(live[CONTROL], live[SYMBOL])
    assert gate == r['cost_gate'] and not gate['passed']
    assert {k for k, v in gate['checks'].items() if not v} == {'work_reduction', 'no_hot_local'}
    old, new = [next(x for x in live[s]['loops'] if x['kind'] == 'integer') for s in (CONTROL, SYMBOL)]
    assert (old['static_instructions'], new['static_instructions']) == (323, 335)
    assert live[SYMBOL]['allocated_gpr'] == 168 and new['max_live_gpr'] == 166
    assert sum(v for op, v in new['opcode_counts'].items() if op.startswith('LDL')) == 4
    assert sum(v for op, v in new['opcode_counts'].items() if op.startswith('STL')) == 1
    assert sum('0xaaab' in op for _, op in instructions(sass, CONTROL, old)) == 2
    assert not any('0xaaab' in op for _, op in instructions(sass, SYMBOL, new))
    assert [trace(sass, s, live[s])['peak_started_not_finished_chains'] for s in (CONTROL, SYMBOL)] == [8, 8]
    assert (B / (STEM + '_generated.cuh')).read_text() == generated_header()
    assert 'Compile gate failed' in (E / 'tmp/o378_v140_codegen.log').read_text()

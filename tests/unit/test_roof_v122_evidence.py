"""Replay raw results and the exact compiled handoff, without CUDA or Torch."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v122'
B=E/'reports/o378_roof_v122_handoff_codegen'
RUN=E/'runs/o378_roof_v122_handoff_full24'
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def test_frozen_files_sources_and_control_are_exact():
    from compare_a100_codegen import compare
    from probe_partial_handoff_codegen import CONTROL,producer_header,inspect_live
    index=read(E/'index.json');receipt=read(B/'codegen.json');env=read(RUN/'environment.json')
    tracked=set(subprocess.check_output(['git','ls-files','-z','--',str(E.relative_to(ROOT))],cwd=ROOT).decode().split('\0'))
    assert len(index['files'])==46
    for f in index['files']:
        p=E/f['path']
        assert p.relative_to(ROOT).as_posix() in tracked
        assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
    for source,digest in receipt['sources'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',index['compile_source_commit']+':'+source],cwd=ROOT)).hexdigest()==digest
    assert receipt['source_commit']==index['compile_source_commit']
    assert env['git_commit']==index['runtime_source_commit']
    assert env['extension_sha256']==index['formal_extension_sha256']
    assert (B/'formal_extension_sha256.txt').read_text().split()[0]==index['formal_extension_sha256']
    for name,digest in {**receipt['artifact_sha256'],**receipt['headers']}.items():assert sha(B/name)==digest
    assert (B/'o78_partial_handoff_producer_generated.cuh').read_text()==producer_header()
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),(B/'o78_partial_handoff.sass').read_text(),'^'+CONTROL+'$')==receipt['control_comparison']
    live=inspect_live((B/'liveness.txt').read_text())
    assert live==receipt['liveness'] and live['registers']==128
    loops={l['kind']:l for l in live['loops']}
    assert loops['producer']['static_instructions']==271 and loops['consumer']['static_instructions']==224
    for role in ('producer','consumer'):
        assert not any(op.startswith(('LDL','STL')) for op in loops[role]['opcode_counts'])
    counts=loops['producer']['opcode_counts']
    assert counts['IMMA.16864.S4.S4']==counts['IMMA.16864.U4.S4']==32
    assert counts['LDSM.16.M88.4']==16 and receipt['compile_gate']['passed']
    ptx=(B/'o78_partial_handoff.ptx').read_text()
    body=next(s for s in re.split(r'(?=\.visible \.entry )',ptx)
        if s.startswith('.visible .entry adangel_roof_o78_partial_handoff_candidate('))
    assert all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32','bar.arrive'))


def test_all24_pairing_raw_events_MSE_and_summary():
    from benchmark_partial_handoff import summarize
    from benchmark_a100_o1 import stats
    result=rows(RUN/'results.jsonl');summary=read(RUN/'summary.json')
    assert len(result)==288 and summarize(result)==summary
    ids={f'layer_{l:02d}_{p}_proj' for l in (0,6,12,18,24,31) for p in ('q','k','v','o')}
    assert {r['sample_id'] for r in result}==ids
    pairs={(r['sample_id'],r['variant'],r['round'],r['policy']):r for r in result}
    for r in result:
        assert len(r['raw_ms'])==200 and min(r['raw_ms'])>0
        assert stats(r['raw_ms'])==pytest.approx(r['summary'])
        assert r['bitwise_best'] and r['finite_fp32'] and r['mode']=='compute_only'
        assert r['reference']=={'o7':'o5','o8':'o6'}[r['variant']]
        mate=pairs[r['sample_id'],r['variant'],r['round'],1-r['policy']]
        assert r['mse_vs_reference']==mate['mse_vs_reference'] and r['guard']==mate['guard']
        assert r['handoff_guard']['wide_integer_ctas']==0
        assert not r['conversion_included'] and r['CPU_narrow_guard_cached']
    assert not summary['production_default_changed'] and not summary['conversion_or_E2E_measured']
    assert summary['no_filtering'] and summary['bitwise_best']
    for s in summary['summary']:
        assert s['records']==72 and s['samples']==24
        if s['policy']==1:assert s['paired_ci95'][1]<1
    observed=rows(RUN/'source_provenance.jsonl')
    authority=rows(ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')
    assert observed==authority
    env=read(RUN/'environment.json')
    assert (env['warmup'],env['repeats'],env['rounds'])==(1000,200,3)
    assert env['resources']['0']['active_warps']==12 and env['resources']['1']['active_warps']==16


def test_sufficient_norm_gate_synthetic_routes_and_sanitizers():
    data=rows(E/'reports/o378_roof_v122_handoff_data_r2/results.jsonl')
    assert len(data)==48 and len({(r['sample_id'],r['variant']) for r in data})==48
    for r in data:
        s=r['statistics']
        assert s['data_gate_passed'] and s['maximum_squared_bound']<=32767**2
        assert s['narrow_integer_ctas']==s['integer_ctas'] and s['wide_integer_ctas']==0
        assert s['narrow_fraction_of_all_ctas']>=.95 and r['v99_source_exact']
    for name in ('validation','memcheck','synccheck','racecheck'):
        v=read(E/f'reports/o378_roof_v122_handoff_{name}/validation.json')
        assert v['passed'] and v['statuses_seen']==[0,1,3] and len(v['checks'])==8
        assert all(c['bitwise_best'] and c['finite_fp32'] and c['nondefault_stream'] for c in v['checks'])
    for name in ('memcheck_r2','synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (E/f'tmp/o378_v122_{name}.log').read_text()
    assert '0 hazards displayed (0 errors, 0 warnings)' in (E/'tmp/o378_v122_racecheck.log').read_text()

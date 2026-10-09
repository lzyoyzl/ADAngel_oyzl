"""Frozen release evidence must support the promoted defaults and report."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/evidence/sm80_production_release_20261009'
RUN=EVIDENCE/'runs/sm80_production_four24'


def test_complete_formal_run_and_binary_audit():
    completion=json.loads((RUN/'completion.json').read_text())
    assert completion==dict(passed=True,records=1728,samples=24,defaults_applied=True)
    env=json.loads((RUN/'environment.json').read_text())
    audit=json.loads((EVIDENCE/'reports/sm80_production_release/audit_r3/audit.json').read_text())
    assert audit['passed'] and audit['extension_sha256']==env['extension_sha256']
    for entry in audit['entries'].values():
        assert entry['candidate_sass_comparison']['passed']
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and entry['no_int8_mma']
    assert 'ERROR SUMMARY: 0 errors' in (EVIDENCE/'reports/sm80_production_release/memcheck.log').read_text()


def test_all_records_and_paired_statistics():
    rows=[json.loads(line) for line in (RUN/'results.jsonl').read_text().splitlines()]
    key=lambda r:(r['sample_id'],r['variant'],r['mode'],r['round'],r['policy'])
    index={key(r):r for r in rows}
    assert len(index)==len(rows)==1728
    ids=sorted({r['sample_id'] for r in rows})
    assert len(ids)==24
    for r in rows:
        assert r['mse_regression_passed'] and math.isfinite(r['mse'])
        assert all(len(v)==200 and all(t>0 and math.isfinite(t) for t in v) for v in r['raw_ms'].values())
        if r['policy']=='production':
            assert r['bitwise_frozen_best'] and r['kernel']['production_default']
            old=index[(*key(r)[:-1],'legacy')]
            assert abs(r['mse']-old['mse'])<=1e-12+1e-5*abs(old['mse'])
    for s in json.loads((RUN/'summary.json').read_text()):
        v,mode,policy=s['variant'],s['mode'],s['policy']
        stage='gemm' if mode=='compute_only' else 'total'
        latency=[statistics.median(index[sid,v,mode,i,policy]['stages'][stage]['median_ms'] for i in range(3)) for sid in ids]
        ratios=[statistics.median(index[sid,v,mode,i,'legacy']['stages'][stage]['median_ms']/index[sid,v,mode,i,policy]['stages'][stage]['median_ms'] for i in range(3)) for sid in ids]
        assert s['median_ms']==statistics.median(latency)
        assert s['paired_speedup']==statistics.median(ratios)
        assert s['samples']==24 and s['records']==72
        if policy=='production':assert s['speedup_ci95'][0]>1

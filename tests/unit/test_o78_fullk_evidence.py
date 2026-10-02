"""Offline checks of measured v67 data; single-shot proof is not performance."""
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/evidence/a100_o378_roof_v67'
RUN=E/'runs/o378_roof_v67_trace24'


def readlines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def test_timing_coverage_and_paired_summary():
    rows=readlines(RUN/'results.jsonl');result=json.loads((RUN/'summary.json').read_text())
    assert len(rows)==288 and result['production_default_changed'] is False
    index={(r['sample_id'],r['variant'],r['round'],r['policy']):r for r in rows}
    ids=sorted({r['sample_id'] for r in rows})
    assert len(ids)==24 and len(index)==288
    assert set(index)=={(s,v,r,p) for s in ids for v in ('o7','o8') for r in range(3) for p in (0,1)}
    for row in rows:
        values=row['raw_ms']; assert len(values)==200 and min(values)>0
        assert math.isclose(statistics.median(values),row['summary']['median_ms'],rel_tol=1e-14)
        mean=statistics.fmean(values);cv=statistics.pstdev(values)/mean*100
        assert math.isclose(cv,row['summary']['cv_percent'],rel_tol=1e-13,abs_tol=1e-13)
        assert row['finite_fp32'] and row['output_close_best'] and row['MSE_regression_passed']
        assert abs(row['mse_vs_paired_fp16']-row['current_best_mse'])<=1e-12+1e-5*abs(row['current_best_mse'])
        if row['policy']==0: assert row['bitwise_equal_best'] and row['mse_vs_best']==0
    for summary in result['records']:
        v,p=summary['variant'],summary['policy']
        med=statistics.median(statistics.median(index[s,v,r,p]['summary']['median_ms'] for r in range(3)) for s in ids)
        speed=statistics.median(statistics.median(index[s,v,r,0]['summary']['median_ms']/index[s,v,r,p]['summary']['median_ms'] for r in range(3)) for s in ids)
        assert med==summary['median_ms'] and speed==summary['paired_speedup']
        assert summary['cv_failed_records']==sum(r['summary']['cv_percent']>=3 for r in rows if r['variant']==v and r['policy']==p)
        errors=[index[s,v,0,p]['mse_vs_paired_fp16'] for s in ids]
        assert statistics.median(errors)==summary['median_mse'] and statistics.mean(errors)==summary['mean_mse']
        if p==1: assert summary['paired_speedup_ci95'][0]>1


def test_real_guard_fallback_and_resources():
    rows=readlines(RUN/'results.jsonl')
    for row in rows:
        g=row['guard'];assert g['ctas']==2048 and g['invalid_ctas']==0
        expected=12 if row['variant']=='o8' and row['sample_id']=='layer_24_o_proj' else 0
        assert g['fallback_ctas']==expected and g['integer_ctas']==2048-expected
        assert g['metadata_bytes']==1089536 and g['preparation_wall_ms']>0
        resource=row['resources']
        assert resource['registers_per_thread']==168 and resource['active_blocks_per_sm']==3
        assert resource['threads']==128 and resource['shared_memory_bytes']==34304
    env=json.loads((RUN/'environment.json').read_text())
    assert env['scope']=='cached_compute_only_not_conversion_or_Cold_or_steady_state'
    assert env['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_payload_binding_and_semantics():
    proof=E/'reports/o378_roof_v67_payload_proof24'
    entries=readlines(proof/'payload_proof.jsonl')
    assert len(entries)==48 and len({(r['sample_id'],r['variant']) for r in entries})==48
    assert all(r['native_integer_payload_and_scales_match_reference_bitwise'] for r in entries)
    for name in ('o378_roof_v67_screen_r1','o378_roof_v67_memcheck','o378_roof_v67_synccheck','o378_roof_v67_payload_proof24'):
        result=json.loads((E/'reports'/name/'validation.json').read_text())
        assert result['passed'] and result['count']==32
        assert len(result['checks'])==32
        assert all(c['bitwise_equal_best'] for c in result['checks'] if c['policy']==0 or c['force_fallback'])
    env=json.loads((proof/'environment.json').read_text())
    assert env['args']['warmup']==0 and env['args']['repeats']==1
    # Explicitly prohibit borrowing this single-shot run for performance claims.
    assert len(readlines(proof/'results.jsonl'))==96


def test_same_entry_isa_and_limited_sanitizers():
    directory=E/'reports/o378_roof_v67_codegen'
    result=json.loads((directory/'codegen.json').read_text())
    assert set(result['entries'])=={'adangel_roof_o78_fullk_control','adangel_roof_o78_fullk_candidate'}
    sass=(directory/'o78_fullk.sass').read_text()
    for name,x in result['entries'].items():
        assert x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma'] and x['all_copies_bypass_l1']
        block=next(b for b in re.split(r'(?=Function\s*:\s*)',sass) if b.startswith('Function : '+name))
        assert re.search(r'IMMA[^;]*\.U4\.S4',block) and re.search(r'IMMA[^;]*\.S4\.S4',block)
        assert not re.search(r'IMMA[^;]*\.[SU]8\.',block)
    log=(directory/'build.log').read_text()
    assert log.count('8 bytes spill stores, 8 bytes spill loads')==2
    for tool in ('memcheck','synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (E/f'reports/o378_roof_v67_{tool}.log').read_text()

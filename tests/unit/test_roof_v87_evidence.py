"""Recompute v87's negative O7 screen, exact INT4 work, and finite validation scope."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v87'
B=E/'reports/o378_roof_v87_codegen'
RUN=E/'runs/o378_roof_v87_screen'


def test_raw_events_complete_pairing_MSE_and_failed_CV_retained():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o7_shift_coefficient import timing_contract
    rows=[json.loads(s) for s in (RUN/'results.jsonl').read_text().splitlines()]
    assert len(rows)==24
    assert {r['sample_id'] for r in rows}=={
        'layer_00_q_proj','layer_00_k_proj','layer_00_v_proj','layer_00_o_proj'}
    assert {r['round'] for r in rows}=={0,1,2}
    for r in rows:
        assert r['variant']=='o7' and r['mode']=='compute_only' and r['paired_fp16']=='o5'
        assert all(r[k] for k in ('finite_fp32','metadata_exact','bitwise_equal_v67','MSE_regression_passed'))
        assert r['mse_vs_v67']==r['max_abs_vs_v67']==0
        assert r['guard']['ctas']==r['guard']['integer_ctas']==2048
        assert r['guard']['fallback_ctas']==r['guard']['invalid_ctas']==0
        assert set(r['raw_ms'])=={'gemm','total'} and r['raw_ms']['gemm']==r['raw_ms']['total']
        for key,value in timing_contract('compute_only',100).items():assert r[key]==value
        for stage,times in r['raw_ms'].items():
            assert len(times)==200 and min(times)>0
            assert stats(times)==pytest.approx(r['stage_summaries'][stage])
    summary=json.loads((RUN/'summary.json').read_text())
    assert summary['records']==summarize(rows,('compute_only',),('o7',))
    assert summary['no_filtering'] and not summary['production_default_changed']
    old,new=summary['records']
    assert old['selected_cv_failed_records']==12 and new['selected_cv_failed_records']==10
    assert old['median_ms']==pytest.approx(.448512) and new['median_ms']==pytest.approx(.461312)
    assert new['paired_speedup']==pytest.approx(.9694805089374593)
    assert new['paired_speedup_ci95'][1]<1
    assert old['median_mse']==new['median_mse'] and old['mean_mse']==new['mean_mse']


def test_same_preparation_and_resource_usage_and_provenance():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    env=json.loads((RUN/'environment.json').read_text())
    assert env['git_commit'].startswith('f9d0e36') and env['variants']==['o7']
    assert env['source_quantization']=='original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope']=='cached_compute_only_not_E2E'
    assert env['no_filtering'] and not env['production_default_changed']
    assert env['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['control']=='v78_eight_chain_same_v73_preparation'
    assert env['candidate']=='v87_O7_coefficient_shift_same_v73_preparation'
    for policy in ('0','1'):
        r=env['resources'][policy]
        assert r['cta_tile']==[64,128,128] and r['pipeline_stages']==2 and r['threads']==128
        assert r['local_size_bytes']==0 and r['registers_per_thread']==168 and r['active_blocks_per_sm']==3
        assert r['shared_memory_bytes']==34304
    for path,digest in env['codegen']['shift_coefficient']['runtime_source_sha256'].items():
        assert sha(ROOT/path)==digest
    preparation=json.loads((ROOT/'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen/build.json').read_text())
    assert env['gpu_preparation_build']==preparation
    sources=[json.loads(s) for s in (RUN/'source_provenance.jsonl').read_text().splitlines()]
    assert len(sources)==len({p['sample_id'] for p in sources})==4
    for p in sources:
        assert p['variant']=='o7' and re.fullmatch('[0-9a-f]{64}',p['raw_sha256'])
        for name in ('activation','weight'):
            assert p[name]['group_size']==128 and p[name]['shape']==[4096,4096]


def test_exact_control_code_and_candidate_native_MMA_and_coefficient_dependencies():
    from compare_a100_codegen import compare
    from inspect_o78_register_liveness import analyze
    from inspect_eight_chain_schedule import trace
    from probe_o7_shift_coefficient_codegen import CONTROL,SYMBOL,generated_header
    from benchmark_o7_shift_coefficient import coefficient_dependencies
    from probe_roof_fullk_integer_codegen import static_entries
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((B/'codegen.json').read_text())
    assert receipt['source_commit'].startswith('c50d508') and receipt['supported_variant']=='o7'
    assert not receipt['production_default_changed']
    for path,digest in receipt['sources'].items():assert sha(ROOT/path)==digest
    for path,digest in receipt['artifact_sha256'].items():assert sha(B/path)==digest
    header=B/'o7_shift_coefficient_generated.cuh'
    assert sha(header)==receipt['generated_header_sha256']
    assert header.read_text()==generated_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    proof=json.loads((B/'coordinate_verification.json').read_text())
    assert proof==receipt['coordinate_verification']==dict(passed=True,threads=128,outputs=8192,
        unique_rows_per_thread=4,gpu_execution=False)
    sass=(B/'o7_shift_coefficient.sass').read_text()
    old=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old,sass,'^'+CONTROL+'$')==receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})==receipt['entries']
    for symbol,n in ((CONTROL,383),(SYMBOL,380)):
        ptx=next(t for t in re.split(r'(?=\.visible \.entry )',(B/'o7_shift_coefficient.ptx').read_text())
            if t.startswith('.visible .entry '+symbol+'('))
        assert all(t in ptx for t in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
        if symbol==SYMBOL:assert 'shf.l.wrap.b32' in ptx
        live=analyze((B/'liveness.txt').read_text(),symbol)
        assert live==receipt['liveness'][symbol] and live['allocated_gpr']==168
        loop=next(x for x in live['loops'] if x['kind']=='integer')
        assert loop['static_instructions']==n and loop['opcode_counts']['LDSM.16.M88.4']==16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128']==10
        assert loop['opcode_counts']['IMMA.16864.S4.S4']==loop['opcode_counts']['IMMA.16864.U4.S4']==32
        assert not any(op.startswith(('LDL','STL')) for op in loop['opcode_counts'])
        assert trace(sass,symbol,live)['total_mma']==64
        entry=receipt['entries'][symbol]
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma'] and entry['all_copies_bypass_l1']
    deps=coefficient_dependencies(sass,SYMBOL,receipt['liveness'][SYMBOL])
    assert len(deps['coefficient_shifts'])==len(deps['coefficient_imad_updates'])==64
    env=json.loads((RUN/'environment.json').read_text())
    assert env['codegen']['shift_coefficient']['dependencies']==deps


def test_synthetic_all_shifts_fallback_O8_rejection_and_finite_sanitizer_scope():
    for path in (RUN,*(E/f'runs/o378_roof_v87_{t}' for t in ('memcheck','synccheck','racecheck'))):
        r=json.loads((path/'validation.json').read_text())
        assert r['passed'] and r['count']==len(r['checks'])==32
        assert r['edge_count']==len(r['edge_checks'])==9
        assert r['scope']=='small_MN_full_K4096_not_4096cubed_sanitizer'
        assert r['o8_rejected_before_launch']
        assert {c['variant'] for c in r['checks']}=={'o7'}
        assert {tuple(c['shape']) for c in r['checks']}=={(64,128,4096),(128,256,4096)}
        assert {c['mode'] for c in r['checks']}=={'conversion_only','compute_only','cold','steady_state'}
        for c in r['checks']:
            assert all(c[k] for k in ('bitwise_equal_v67','finite_fp32','semantic_tolerance_passed',
                'metadata_exact','nondefault_stream'))
            if c['pattern']=='wide_scale':assert c['fallback_ctas']==c['ctas'] and c['integer_ctas']==0
        assert all(c['metadata_exact'] for c in r['edge_checks'])
        assert r['legal_shift_count']==len(r['legal_shift_checks'])==31
        assert [c['shift'] for c in r['legal_shift_checks']]==list(range(31))
        for c in r['legal_shift_checks']:
            assert c['integer_ctas']==c['ctas'] and c['fallback_ctas']==c['invalid_ctas']==0
            assert c['policies']==[0,1,2] and c['exact_analytic_fp32']
    for tool in ('memcheck','synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B/f'{tool}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B/'racecheck.log').read_text()

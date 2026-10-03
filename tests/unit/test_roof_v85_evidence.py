"""Recompute the register-layout negative screen; do not hide mapping or CV failures."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'python'))
E=ROOT/'docs/evidence/a100_o378_roof_v85'
B=E/'reports/o378_roof_v85_codegen_checked'
RUN=E/'runs/o378_roof_v85_screen_checked'


def test_raw_events_negative_speedups_and_unchanged_mse():
    from benchmark_a100_o1 import stats
    from benchmark_o78_coefficient_probe import summarize
    from benchmark_o78_register_layout import timing_contract

    rows=[json.loads(s) for s in (RUN/'results.jsonl').read_text().splitlines()]
    assert len(rows)==48
    assert {r['sample_id'] for r in rows}=={
        'layer_00_q_proj','layer_00_k_proj','layer_00_v_proj','layer_00_o_proj'}
    assert {r['round'] for r in rows}=={0,1,2}
    for r in rows:
        assert r['mode']=='compute_only'
        assert r['paired_fp16']=={'o7':'o5','o8':'o6'}[r['variant']]
        assert all(r[k] for k in ('finite_fp32','metadata_exact','bitwise_equal_v67','MSE_regression_passed'))
        assert r['mse_vs_v67']==r['max_abs_vs_v67']==0
        assert r['guard']['ctas']==r['guard']['integer_ctas']==2048
        assert r['guard']['fallback_ctas']==r['guard']['invalid_ctas']==0
        assert set(r['raw_ms'])=={'gemm','total'}
        assert r['raw_ms']['gemm']==r['raw_ms']['total']
        for key,value in timing_contract('compute_only',100).items():assert r[key]==value
        for stage,times in r['raw_ms'].items():
            assert len(times)==200 and min(times)>0
            assert stats(times)==pytest.approx(r['stage_summaries'][stage])
    saved=json.loads((RUN/'summary.json').read_text())
    assert saved['no_filtering'] and not saved['production_default_changed']
    assert saved['records']==summarize(rows,('compute_only',))
    for variant,failures in (('o7',12),('o8',9)):
        old,new=[r for r in saved['records'] if r['variant']==variant]
        assert old['median_mse']==new['median_mse'] and old['mean_mse']==new['mean_mse']
        assert old['selected_cv_failed_records']==12 and new['selected_cv_failed_records']==failures
        assert new['paired_speedup']<1 and new['paired_speedup_ci95'][1]<1


def test_preparation_cost_provenance_and_actual_resource_usage():
    env=json.loads((RUN/'environment.json').read_text())
    assert env['source_quantization']=='original_FP16_direct_source_quantization_excluded'
    assert env['timing_scope']=='cached_compute_only_not_E2E'
    assert env['no_filtering'] and not env['production_default_changed']
    assert env['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
    assert env['control']=='v78_LDSM_same_v73_preparation'
    assert env['candidate']=='v85_register_layout_with_online_repack'
    for policy,local in (('0',0),('1',8)):
        r=env['resources'][policy]
        assert r['cta_tile']==[64,128,128] and r['pipeline_stages']==2
        assert r['threads']==128 and r['local_size_bytes']==local
        assert r['registers_per_thread']==168 and r['active_blocks_per_sm']==3
        assert r['shared_memory_bytes']==34304
    provenance=[json.loads(s) for s in (RUN/'source_provenance.jsonl').read_text().splitlines()]
    assert len(provenance)==8 and len({(p['sample_id'],p['variant']) for p in provenance})==8
    for p in provenance:
        assert re.fullmatch('[0-9a-f]{64}',p['raw_sha256'])
        for name in ('activation','weight'):
            assert p[name]['group_size']==128 and p[name]['shape']==[4096,4096]
    from probe_o78_register_layout_codegen import generated_driver
    driver=(B/'register_layout_driver.cu').read_text()
    assert driver==generated_driver((ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text())
    assert driver.count('o78_register_pack::pack<')==2


def test_coordinate_proof_same_entry_int4_and_unchanged_control():
    from compare_a100_codegen import compare
    from inspect_o78_register_liveness import analyze
    from probe_o78_register_layout_codegen import CONTROL,SYMBOL,generated_header
    from probe_roof_fullk_integer_codegen import static_entries

    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((B/'codegen.json').read_text())
    assert receipt['source_commit'].startswith('258e053')
    assert not receipt['production_default_changed']
    for path,digest in receipt['sources'].items():assert sha(ROOT/path)==digest
    for path,digest in receipt['artifact_sha256'].items():assert sha(B/path)==digest
    assert sha(B/'o78_register_layout_generated.cuh')==receipt['generated_header_sha256']
    assert (B/'o78_register_layout_generated.cuh').read_text()==generated_header(
        (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    proof=json.loads((B/'mapping_verification.json').read_text())
    assert proof==receipt['coordinate_verification']==dict(passed=True,a_register_nibbles=16384,
        b_register_nibbles=32768,legacy_adjacent_b_mismatches=24576,gpu_execution=False)
    sass=(B/'o78_register_layout.sass').read_text()
    old=(ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass').read_text()
    assert compare(old,sass,'^'+CONTROL+'$')==receipt['control_comparison']
    assert receipt['control_comparison']['passed']
    assert static_entries(sass,'^adangel_roof_o78_(?:eight_chain_candidate|register_layout_candidate)$',
        {CONTROL,SYMBOL})==receipt['entries']
    ptx=next(t for t in re.split(r'(?=\.visible \.entry )',(B/'o78_register_layout.ptx').read_text())
        if t.startswith('.visible .entry '+SYMBOL+'('))
    assert all(t in ptx for t in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32',
        'ld.shared.v4.b32'))
    for symbol,count,load in ((CONTROL,383,'LDSM.16.M88.4'),(SYMBOL,379,'LDS.128')):
        actual=analyze((B/'liveness.txt').read_text(),symbol)
        assert actual==receipt['liveness'][symbol] and actual['allocated_gpr']==168
        loop=next(x for x in actual['loops'] if x['kind']=='integer')
        assert loop['static_instructions']==count and loop['opcode_counts'][load]==16
        assert loop['opcode_counts']['LDGSTS.E.BYPASS.128']==10
        assert loop['opcode_counts']['IMMA.16864.S4.S4']==loop['opcode_counts']['IMMA.16864.U4.S4']==32
        assert not any(op.startswith(('LDL','STL')) for op in loop['opcode_counts'])
        if symbol==SYMBOL:assert not any(op.startswith('LDSM') for op in loop['opcode_counts'])
        e=receipt['entries'][symbol]
        assert e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
    fb=next(x for x in receipt['liveness'][SYMBOL]['loops'] if x['kind']=='fp32_fallback')
    assert fb['opcode_counts']['LDL']==2


def test_finite_gpu_and_sanitizer_scope_is_not_full_trace_coverage():
    for path in (RUN,*(E/f'reports/o378_roof_v85_{t}' for t in ('memcheck','synccheck','racecheck'))):
        r=json.loads((path/'validation.json').read_text())
        assert r['passed'] and r['count']==len(r['checks'])==64
        assert r['edge_count']==len(r['edge_checks'])==12
        assert r['scope']=='small_MN_full_K4096_not_4096cubed_sanitizer'
        assert {tuple(c['shape']) for c in r['checks']}=={(64,128,4096),(128,256,4096)}
        assert {c['mode'] for c in r['checks']}=={'conversion_only','compute_only','cold','steady_state'}
        for c in r['checks']:
            assert all(c[k] for k in ('bitwise_equal_v67','finite_fp32','semantic_tolerance_passed',
                'metadata_exact','nondefault_stream'))
            assert c['invalid_ctas']==0
            if c['pattern']=='wide_scale':assert c['fallback_ctas']==c['ctas'] and c['integer_ctas']==0
        assert all(c['metadata_exact'] for c in r['edge_checks'])
        assert sum(c['invalid_ctas'] for c in r['edge_checks'])==3
    for tool in ('memcheck','synccheck'):
        assert 'ERROR SUMMARY: 0 errors' in (B/f'{tool}.log').read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in (B/'racecheck.log').read_text()

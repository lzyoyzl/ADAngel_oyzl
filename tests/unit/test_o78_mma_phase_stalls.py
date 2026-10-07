"""Exact consumer-role accounting is not causal attribution or a new run."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_o78_mma_phase_stalls import CODEGEN,CAPTURE,SYMBOL,analyze_capture,normalized


@pytest.fixture(scope='module')
def inputs():
    return ((CODEGEN/'o78_eight_chain.sass').read_text(),(CODEGEN/'liveness.txt').read_text(),
        (CAPTURE/'o8_warm_source_sass.csv').read_text(),json.loads((CAPTURE/'analysis.json').read_text()))


def test_role_and_sampling_closure(inputs):
    r=analyze_capture(*inputs);c=r['consumer_categories'];s=r['integer_loop_static']
    assert sum(s.values())==383 and s.get('factor_product',0)==0
    assert s['partial_times_activation_factor']==s['weighted_integer_accumulate']==s['high_times_16']==64
    assert sum(x['static_instructions'] for x in c.values())==1976
    assert sum(x['warp_instructions'] for x in c.values())==105521152
    assert sum(x['not_issued_samples'] for x in c.values())==r['not_issued_samples']==10557
    mma=[v for k,v in c.items() if k.startswith('mma_')]
    assert sum(v['reason_samples']['wait'] for v in mma)==1908
    assert sum(v['reason_samples']['math'] for v in mma)==2614
    assert sum(v['reason_samples']['short_sb'] for v in mma)==505
    scale=[c[k] for k in ('partial_times_activation_factor','weighted_integer_accumulate')]
    assert sum(v['reason_samples']['wait'] for v in scale)==854
    assert c['loop_exit_control']['reason_samples']['barrier']==1239
    assert r['calls'][0]['warp_instructions']==262144
    # One active CALL per warp, not one active CALL on each of32 groups.
    assert r['calls'][0]['predicated_on_thread_instructions']//32==2048*4
    assert not any(r[k] for k in ('new_GPU_capture','new_performance_or_MSE_result','production_default_changed'))


def test_source_operands_predicates_and_counts_are_not_guessed(inputs):
    sass,live,source,prior=inputs
    for bad in (source.replace('MOV R1, c[0x0][0x28]','MOV R2, c[0x0][0x28]',1),
                source.replace('@!P1  CALL.REL.NOINC','@P1   CALL.REL.NOINC',1)):
        with pytest.raises(ValueError,match='instruction mismatch'):analyze_capture(sass,live,bad,prior)
    wrong=json.loads(json.dumps(prior));wrong['pc_sampling']['reason_samples']['wait']+=1
    with pytest.raises(ValueError,match='sampling totals'):analyze_capture(sass,live,source,wrong)
    assert normalized('LDGSTS.E.BYPASS.128 [R4+0x200], [R6.64]')==normalized('LDGSTS.E.BYPASS.128 [R4+0x200][R6.64]')
    assert normalized('@P1 BRA 0x100')==normalized('@P1 BRA 0x900')
    assert normalized('@P1 BRA 0x100')!=normalized('@!P1 BRA 0x100')


def test_frozen_analysis_replays_and_inputs_match(inputs):
    path=ROOT/'docs/evidence/a100_o378_roof_v133/analysis.json'
    frozen=json.loads(path.read_text());r=analyze_capture(*inputs)
    for key,value in r.items():assert frozen[key]==value
    for name,digest in frozen['inputs'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'scripts/analyze_o78_mma_phase_stalls.py').read_bytes()).hexdigest()==frozen['script_sha256']

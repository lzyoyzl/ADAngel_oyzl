"""Immutable full24 v92 first/repeat evidence, actual encoded native kernels."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze_sixteen_chain_evidence import read_run, compare as compare_runs
from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from probe_roof_fullk_integer_codegen import static_entries
from probe_sixteen_chain_codegen import CONFIG, generated_header, worth_runtime

E=ROOT/'docs/evidence/a100_o378_roof_v92'


@pytest.mark.parametrize('kind',('o3','o78'))
def test_actual_more_chains_same_native_work_and_exact_encoded_control(kind):
    cfg=CONFIG[kind];d=E/f'reports/o378_roof_v92_{kind}_codegen'
    receipt=json.loads((d/'codegen.json').read_text())
    for name,sha in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'): continue  # Complete binary retained outside Git.
        assert hashlib.sha256((d/name).read_bytes()).hexdigest()==sha
    for name,sha in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    assert (d/(cfg['stem']+'_generated.cuh')).read_text()==generated_header(kind)
    symbols={cfg['symbol'],cfg['control']};sass=(d/(cfg['stem']+'.sass')).read_text()
    assert static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)==receipt['entries']
    old=ROOT/('docs/evidence/a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen/o3_grouped_cta.sass'
        if kind=='o3' else 'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass')
    assert compare(old.read_text(),sass,'^'+cfg['control']+'$')['passed']
    for symbol,peak in ((cfg['control'],8),(cfg['symbol'],16)):
        live=receipt['liveness'][symbol]
        schedule=trace(sass,symbol,live)
        assert schedule['total_mma']==64 and schedule['chains_per_group']==16
        assert schedule['peak_started_not_finished_chains']==peak
    candidate=receipt['liveness'][cfg['symbol']]
    assert worth_runtime(candidate) and receipt['worth_runtime_validation']
    assert candidate['allocated_gpr']==(214 if kind=='o3' else 230)
    assert not receipt['changed_semantics'] and not receipt['production_default_changed']


@pytest.mark.parametrize('kind',('o3','o78'))
def test_all24_raw_statistics_bitwise_MSE_and_retained_retry(kind):
    runs=[read_run(E/f'runs/o378_roof_v92_{kind}_trace24{suffix}')
        for suffix in ('','_retry')]
    for run in runs:
        assert run['records']==(144 if kind=='o3' else 288)
        assert run['identity']['counts']==dict(samples=24,rounds=3,warmup=1000,repeats=200,inner=100)
        assert run['identity']['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
        assert run['no_filtering'] and not run['new_conversion_or_E2E_result']
        # Recomputed means/medians/CV and complete coverage are checked in read_run.
        assert all(r['bitwise_previous_best'] for r in run['summary'])
        assert all(r['paired_speedup_ci95'][1]<1 for r in run['summary'] if r['policy']==1)
    comparison=compare_runs(*runs)
    assert len(comparison)==(2 if kind=='o3' else 4)
    analysis=json.loads((E/'reports/o378_roof_v92_analysis.json').read_text())
    assert analysis[kind+'_repeat_comparison']==comparison
    assert not analysis['production_default_changed'] and analysis['no_filtering']


def test_real_runtime_resources_guard_and_numerical_validation():
    for kind in ('o3','o78'):
        for suffix in ('','_retry'):
            d=E/f'runs/o378_roof_v92_{kind}_trace24{suffix}'
            validation=json.loads((d/'validation.json').read_text())
            assert validation['passed']
            if kind=='o3':
                assert len(validation['checks'])==96 and len(validation['rejected'])==8
                resources=json.loads((d/'resources.json').read_text())
                old,new=resources['v89'],resources['v92']
            else:
                assert validation['count']==64 and validation['edge_count']==12
                assert validation['grouped_coordinate_count']==8
                env=json.loads((d/'environment.json').read_text())
                old,new=env['resources']['0'],env['resources']['1']
                rows=[json.loads(s) for s in (d/'results.jsonl').read_text().splitlines()]
                fallback={(r['sample_id'],r['variant'],r['guard']['fallback_ctas'])
                    for r in rows if r['guard']['fallback_ctas']}
                assert fallback=={('layer_24_o_proj','o8',12)}
            assert old['active_blocks_per_sm']==3 and new['active_blocks_per_sm']==2
            assert new['local_size_bytes']==0

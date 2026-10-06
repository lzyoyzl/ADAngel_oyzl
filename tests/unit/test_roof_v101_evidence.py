import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from summarize_cta_timeline import summarize
from compare_a100_codegen import compare

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v101/reports'


def test_raw_nine_captures_reproduce_canonical_summary():
    actual=summarize(EVIDENCE)
    assert actual==json.loads((EVIDENCE/'analysis.json').read_text())
    assert len(actual['results'])==3
    for row in actual['results']:
        assert row['captures']==3 and row['bitwise_previous_best']
        assert row['output_MSE_vs_previous_best']==0
        assert 88<row['three_CTA_SM_time_fraction_percent']<92
        assert 4<row['full_missing_capacity_equivalent_percent']<6
    assert not actual['new_speedup_claim'] and not actual['production_default_changed']


def test_original_controls_remain_encoded_identical():
    for kind,oldfolder,oldname,symbol in (
            ('o3','a100_o378_roof_v89/reports/o378_roof_v89_o3_codegen','o3_grouped_cta',
             'adangel_roof_o3_grouped_cta_candidate'),
            ('o78','a100_o378_roof_v78/reports/o378_roof_v78_codegen','o78_eight_chain',
             'adangel_roof_o78_eight_chain_candidate')):
        before=ROOT/'docs/evidence'/oldfolder/(oldname+'.sass')
        after=EVIDENCE/f'o378_roof_v101_{kind}_codegen'/(kind+'_cta_timeline.sass')
        assert compare(before.read_text(),after.read_text(),'^'+symbol+'$')['passed']


def test_production_extension_unchanged_and_extra_local_is_disclosed():
    report=json.loads((EVIDENCE/'analysis.json').read_text())
    rows={r['variant']:r for r in report['results']}
    for r in rows.values():
        assert r['extension_sha256']=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'
        assert r['hot_local_instructions']==0
    for v in ('o7','o8'):
        assert rows[v]['resources']['control']['local_bytes']==0
        assert rows[v]['resources']['instrumented']['local_bytes']==8

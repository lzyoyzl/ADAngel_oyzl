from pathlib import Path
import hashlib
import json
import re
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from inspect_interleaved_tail_equivalence import equivalence
from compare_a100_codegen import compare
from probe_interleaved_tail_codegen import CONFIG, generated_header, ordering_summary, worth_runtime
from probe_roof_fullk_integer_codegen import static_entries

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v96/reports'


@pytest.mark.parametrize('kind',('o3','o78'))
def test_raw_compile_identity_exact_entries_and_rejected_order(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/f'o378_roof_v96_{kind}_codegen'
    #Curated Git evidence intentionally excludes cubin; do NOT weaken the
    #runtime checked() contract to accept missing binaries. Verify all source
    #and text artifacts explicitly; raw binaries stay in the SHA-checked tar.
    receipt=json.loads((directory/'codegen.json').read_text())
    for name,digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            assert digest==receipt['cubin_sha256']
            continue
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text()==generated_header(kind)
    sass=(directory/(cfg['stem']+'.sass')).read_text()
    live=(directory/'liveness.txt').read_text()
    assert receipt['source_commit']=='54dbe9c1bc85f8b54aca446ab5d6a11cf9b8dfd5'
    assert receipt['control_comparison']['passed']
    old_evidence='a100_o378_roof_v89' if kind=='o3' else 'a100_o378_roof_v78'
    old=ROOT/'docs/evidence'/old_evidence/cfg['baseline']/(cfg['old_stem']+'.sass')
    assert compare(old.read_text(),sass,'^'+re.escape(cfg['control'])+'$')==receipt['control_comparison']
    symbols={cfg['control'],cfg['symbol']}
    assert static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)==receipt['entries']
    assert not receipt['worth_runtime_validation']
    for symbol in (cfg['control'],cfg['symbol']):
        assert analyze(live,symbol)==receipt['liveness'][symbol]
        schedule=trace(sass,symbol,receipt['liveness'][symbol])
        assert json.loads(json.dumps(schedule))==receipt['schedules'][symbol]
        order=ordering_summary(schedule)
        assert order==receipt['ordering'][symbol]
        assert order['first_next_slice_mma_ordinal']==33
        assert not order['next_slice_starts_before_all_old_mma_finish']
        assert order['peak_started_not_finished_chains']==8
    assert not worth_runtime(receipt['liveness'][cfg['symbol']],receipt['ordering'][cfg['symbol']])


@pytest.mark.parametrize('kind',('o3','o78'))
def test_no_runtime_claims_and_no_false_binary_equivalence(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/f'o378_roof_v96_{kind}_codegen'
    r=json.loads((EVIDENCE/f'o378_roof_v96_{kind}_equivalence.json').read_text())
    actual=equivalence((directory/(cfg['stem']+'.sass')).read_text(),cfg['control'],cfg['symbol'])
    assert all(r[key]==value for key,value in actual.items())
    assert not r['passed'] and not r['new_runtime_measurement'] and not r['production_default_changed']
    assert r['sass_sha256']==hashlib.sha256((directory/(cfg['stem']+'.sass')).read_bytes()).hexdigest()
    assert r['codegen_sha256']==hashlib.sha256((directory/'codegen.json').read_bytes()).hexdigest()
    assert r['script_sha256']==hashlib.sha256((ROOT/'scripts/inspect_interleaved_tail_equivalence.py').read_bytes()).hexdigest()
    doc=(ROOT/'docs/o3_o7_o8_interleaved_tail_20261006.md').read_text()
    assert '不启动 GPU 性能/MSE 测试' in doc and '不是本轮新测' in doc

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
from probe_tail_lookahead_codegen import CONFIG, generated_header, schedule_summary, worth_runtime
from probe_roof_fullk_integer_codegen import static_entries

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v97/reports'


@pytest.mark.parametrize('kind',('o3','o78'))
def test_raw_compile_identity_same_native_work_and_rejected_schedule(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/f'o378_roof_v97_{kind}_codegen'
    receipt=json.loads((directory/'codegen.json').read_text())
    for name,digest in receipt['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for name,digest in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            assert digest==receipt['cubin_sha256']
            continue #Raw binaries are in the SHA-checked tar, not curated Git evidence.
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text()==generated_header(kind)
    assert receipt['source_commit']=='8eeeb04e6e5e76ef3a000c10f826501e5bdaef4b'
    assert receipt['control_comparison']['passed'] and not receipt['worth_runtime_validation']
    assert not receipt['changed_semantics'] and not receipt['production_default_changed']
    sass=(directory/(cfg['stem']+'.sass')).read_text();live=(directory/'liveness.txt').read_text()
    old_evidence='a100_o378_roof_v89' if kind=='o3' else 'a100_o378_roof_v78'
    old=ROOT/'docs/evidence'/old_evidence/cfg['baseline']/(cfg['old_stem']+'.sass')
    assert compare(old.read_text(),sass,'^'+re.escape(cfg['control'])+'$')==receipt['control_comparison']
    symbols={cfg['control'],cfg['symbol']}
    assert static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)==receipt['entries']
    for symbol in symbols:
        actual=analyze(live,symbol)
        assert actual==receipt['liveness'][symbol] and actual['allocated_gpr']==168
        schedule=trace(sass,symbol,actual)
        assert json.loads(json.dumps(schedule))==receipt['schedules'][symbol]
        order=schedule_summary(schedule)
        assert order==receipt['ordering'][symbol]
        assert order['ninth_started_chain_first_mma_ordinal']==33
        assert order['peak_started_not_finished_chains']==8
        loop=next(x for x in actual['loops'] if x['kind']=='integer')
        assert loop['opcode_counts']['IMMA.16864.S4.S4']==loop['opcode_counts']['IMMA.16864.U4.S4']==32
        assert loop['opcode_counts']['LDSM.16.M88.4']==16
    assert not worth_runtime(receipt['liveness'][cfg['symbol']],receipt['ordering'][cfg['symbol']])
    candidate=next(x for x in receipt['liveness'][cfg['symbol']]['loops'] if x['kind']=='integer')
    assert candidate['static_instructions']==(323 if kind=='o3' else 377)
    assert candidate['max_live_gpr']==(166 if kind=='o3' else 164)


@pytest.mark.parametrize('kind',('o3','o78'))
def test_exact_encoding_is_not_misreported_as_a_runtime_result(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/f'o378_roof_v97_{kind}_codegen'
    r=json.loads((EVIDENCE/f'o378_roof_v97_{kind}_equivalence.json').read_text())
    actual=equivalence((directory/(cfg['stem']+'.sass')).read_text(),cfg['control'],cfg['symbol'])
    assert all(r[key]==value for key,value in actual.items())
    assert not r['passed'] and not r['new_runtime_measurement'] and not r['production_default_changed']
    assert r['sass_sha256']==hashlib.sha256((directory/(cfg['stem']+'.sass')).read_bytes()).hexdigest()
    assert r['codegen_sha256']==hashlib.sha256((directory/'codegen.json').read_bytes()).hexdigest()
    assert r['script_sha256']==hashlib.sha256((ROOT/'scripts/inspect_tail_lookahead_equivalence.py').read_bytes()).hexdigest()
    doc=(ROOT/'docs/o3_o7_o8_tail_lookahead_20261006.md').read_text()
    assert '没有候选 kernel launch' in doc and '不是本轮新测' in doc

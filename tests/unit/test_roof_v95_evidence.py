import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from probe_slot_pipeline_codegen import CONFIG, generated_header, worth_runtime

EVIDENCE=ROOT/'docs/evidence/a100_o378_roof_v95/reports'


@pytest.mark.parametrize('kind',('o3','o78'))
def test_bound_compiler_receipt_generated_source_and_raw_text(kind):
    cfg=CONFIG[kind];directory=EVIDENCE/f'o378_roof_v95_{kind}_codegen'
    r=json.loads((directory/'codegen.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for name,digest in r['sources'].items():assert sha(ROOT/name)==digest
    for name,digest in r['artifact_sha256'].items():
        if name.endswith('.cubin'):continue # full hash-bound binary stays in archive
        assert sha(directory/name)==digest
    assert (directory/(cfg['stem']+'_generated.cuh')).read_text()==generated_header(kind)
    assert r['control_comparison']['passed'] and not r['changed_semantics']
    assert not r['production_default_changed'] and not r['worth_runtime_validation']
    assert not worth_runtime(r['liveness'][cfg['symbol']])
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
               for e in r['entries'].values())
    ptx=(directory/(cfg['stem']+'.ptx')).read_text()
    body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+cfg['symbol']+'('))
    assert 'cp.async.mbarrier.arrive.noinc.shared.b64' in body
    assert 'mbarrier.test_wait.parity.shared.b64' in body


def test_actual_resource_query_not_performance_or_correctness():
    r=json.loads((EVIDENCE/'o378_v95_resources.json').read_text())
    assert r['kind']=='CUDA_driver_resource_query_no_candidate_kernel_launch'
    assert r['script_sha256']==hashlib.sha256((ROOT/'scripts/inspect_slot_pipeline_resources.py').read_bytes()).hexdigest()
    for kind,cfg in CONFIG.items():
        q=r['kernels'][kind]
        assert (q['registers'],q['threads'],q['active_blocks_per_sm'])==(168,128,3)
        assert q['local_bytes']==(72 if kind=='o3' else 120)
        assert not q['compile_gate'] and q['shared_bytes']==cfg['shared']
        receipt=json.loads((EVIDENCE/f'o378_roof_v95_{kind}_codegen/codegen.json').read_text())
        assert q['cubin_sha256']==receipt['cubin_sha256']
    assert not list(EVIDENCE.rglob('results.jsonl'))
    assert '84 passed' in (EVIDENCE/'o378_v95_tests.log').read_text()
    assert '94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462' in (
        EVIDENCE/'o378_v95_extension.sha256').read_text()


def test_rejection_follows_measured_static_work_not_invented_timing():
    for kind,instructions,loads in (('o3',436,15),('o78',526,30)):
        r=json.loads((EVIDENCE/f'o378_roof_v95_{kind}_codegen/codegen.json').read_text())
        loop=next(l for l in r['liveness'][CONFIG[kind]['symbol']]['loops'] if l['kind']=='integer')
        assert loop['static_instructions']==instructions and loop['opcode_counts']['LDL']==loads
        assert loop['opcode_counts']['LDSM.16.M88.4']==16
        assert loop['opcode_counts']['IMMA.16864.S4.S4']==loop['opcode_counts']['IMMA.16864.U4.S4']==32
        assert not any(k.startswith('BAR.') for k in loop['opcode_counts'])

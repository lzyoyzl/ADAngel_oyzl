"""Replay frozen compile evidence; deliberately no GPU/performance claims."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import probe_register_mma_codegen as probe
from compare_a100_codegen import compare,instructions
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze

E=ROOT/'docs/evidence/a100_o378_roof_v119'
D=E/'reports/o378_roof_v119_codegen'
R=json.loads((D/'codegen.json').read_text())
INDEX=json.loads((E/'index.json').read_text())


def sha(data):return hashlib.sha256(data).hexdigest()


def test_original_file_hashes_source_commit_and_no_new_performance_scope():
    assert INDEX['source_commit']==R['source_commit']=='e92177f5cb8c65f7bfb9fcb38ab1b9f626c592e1'
    assert not INDEX['production_default_changed'] and not INDEX['new_performance_or_MSE']
    assert len(INDEX['files'])==15
    for f in INDEX['files']:
        payload=(E/f['path']).read_bytes()
        assert sha(payload)==f['sha256'] and len(payload)==f['bytes']
    for name,digest in R['sources'].items():
        payload=subprocess.check_output(['git','show',R['source_commit']+':'+name],cwd=ROOT)
        assert sha(payload)==digest,name
    for name,digest in (R['artifact_sha256']|R['generated_sha256']).items():
        assert sha((D/name).read_bytes())==digest,name
    assert not (D/'o78_register_mma.cubin').exists()
    formal=(D/'formal_extension_sha256.txt').read_text().split()[0]
    assert formal=='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def test_generated_math_is_original_and_only_future_notice_retention_changed():
    arch=(ROOT/'third_party/cutlass-src/include/cute/arch/mma_sm80.hpp').read_bytes()
    canonical=subprocess.check_output(['git','-C',str(ROOT/'third_party/cutlass-src'),
        'show',R['cutlass_commit']+':include/cute/arch/mma_sm80.hpp'])
    # The Windows checkout uses CRLF; verify exact canonical pinned Git bytes,
    # not a broad whitespace-normalized source comparison or a relaxed SHA.
    assert sha(canonical)==R['upstream_arch_sha256']
    assert arch.replace(b'\r\n',b'\n')==canonical
    arch_text=canonical.decode()
    notice=arch_text[:arch_text.index('#pragma once')]
    # The actual probe predates the notice-retention fix. Keep its raw header
    # untouched; adjacent NVIDIA_LICENSE covers redistribution. No recompile.
    assert probe.generated_atoms(arch_text).removeprefix(notice)==(D/'register_mma_generated.cuh').read_text()
    body=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    assert probe.generated_body(body)==(D/'o78_register_mma_generated.cuh').read_text()
    assert 'SPDX-License-Identifier: BSD-3-Clause' in (E/'NVIDIA_LICENSE.txt').read_text()
    assert (D/'server_head.txt').read_text().strip()==R['source_commit']
    assert (D/'cutlass_commit.txt').read_text().strip()==R['cutlass_commit']


def test_replay_same_complete_encoded_sass_liveness_and_failed_gate():
    sass=(D/'o78_register_mma.sass').read_text()
    live_text=(D/'liveness.txt').read_text()
    control=probe.CONTROL;candidate=probe.SYMBOL
    old=analyze(live_text,control);new=analyze(live_text,candidate)
    assert old==R['liveness'][control] and new==R['liveness'][candidate]
    assert old['loops']==new['loops']
    assert old['allocated_gpr']==new['allocated_gpr']==168
    loop=next(x for x in new['loops'] if x['kind']=='integer')
    assert loop['static_instructions']==383 and loop['max_live_gpr']==166
    assert not any(x.startswith(('LDL','STL')) for x in loop['opcode_counts'])
    words=instructions(sass,'^(?:'+control+'|'+candidate+')$')
    assert words[control]==words[candidate]
    baseline=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(baseline.read_text(),sass,'^'+control+'$')==R['control_comparison']
    gate=probe.gate(old,new)
    assert not gate['passed'] and not gate['checks']['meaningful_potential']
    for key,value in gate.items():assert R['cost_gate'][key]==value
    assert R['cost_gate']['control_encoding_unchanged'] and R['cost_gate']['full_entry_encoded_equal']
    for symbol in (control,candidate):
        # JSON normalization accounts only for histogram integer-key encoding.
        actual=json.loads(json.dumps(trace(sass,symbol,R['liveness'][symbol])))
        assert actual==R['schedule'][symbol]
        assert actual['total_mma']==64 and actual['peak_started_not_finished_chains']==8

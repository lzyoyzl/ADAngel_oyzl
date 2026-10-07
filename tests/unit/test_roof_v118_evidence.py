"""CPU replay of frozen packed NVFP4 compilation and full24 Event evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
E=ROOT/'docs/evidence/a100_o378_roof_v118'
B=E/'reports/o378_roof_v118_codegen'
RUN=E/'runs/o378_v118_full24'
EXTENSION='94ad3751657474b6c895c32f824554b92951c0cbccd137a0f19d35bb51d7c462'


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_raw_artifacts_sources_and_unchanged_formal_extension():
    index=read(E/'index.json');build=read(B/'build.json');env=read(RUN/'environment.json')
    assert index['artifact_count']==len(index['artifacts'])
    for name,digest in index['artifacts'].items():assert sha(E/name)==digest,name
    for name in index['binary_files_not_committed']:assert not (E/name).exists()
    assert not index['production_default_changed'] and not index['GEMM_modified']
    assert env['extension_sha256']==index['formal_extension_sha256_after']==EXTENSION
    assert build['source_commit']==index['compile_source_commit']
    assert env['git_commit']==index['runtime_source_commit']
    for name,digest in build['sources'].items():
        source=subprocess.check_output(['git','show',index['compile_source_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==digest,name
    source=subprocess.check_output(['git','show',index['runtime_source_commit']+
        ':scripts/benchmark_o7_nv4_swar.py'],cwd=ROOT)
    assert hashlib.sha256(source).hexdigest()==env['codegen']['runtime_source_sha256']


def test_replay_actual_cost_gate_and_verbatim_v73_body(monkeypatch):
    import probe_nv4_swar_codegen as probe
    baseline=ROOT/'docs/evidence/a100_o378_roof_v73/reports/o378_roof_v73_codegen'
    monkeypatch.setattr(probe,'BASELINE',baseline)
    build=read(B/'build.json');audit=probe.audit(B)
    assert audit==build['audit'] and audit['worth_runtime_validation']
    assert audit['control']['instructions']==392 and audit['candidate']['instructions']==320
    assert audit['control']['registers']==31 and audit['candidate']['registers']==30
    assert audit['static_instruction_reduction_fraction']==pytest.approx(72/392)
    assert audit['old_controls']['passed'] and audit['local_instructions']==0
    assert audit['cta_barriers']==1 and audit['candidate']['shared']==256
    assert audit['candidate']['opcode_counts']['POPC']==10
    assert (B/'nv4_swar_generated.cuh').read_text()==probe.generated_header()
    assert (B/'nv4_swar_prepare.cu').read_text()==probe.generated_host()
    for name,digest in build['artifact_sha256'].items():
        if name!='libo78_gpu_prepare.so':assert sha(B/name)==digest,name


def test_all24_four_modes_raw_Event_stats_output_MSE_and_source_identity():
    from analyze_nv4_swar import analyze
    from benchmark_o7_nv4_swar import timing_contract
    data=rows(RUN/'results.jsonl');analysis=analyze(RUN)
    assert analysis==read(RUN/'analysis.json')
    ids={f'layer_{layer:02d}_{proj}_proj' for layer in (0,6,12,18,24,31) for proj in ('q','k','v','o')}
    assert len(data)==576 and {r['sample_id'] for r in data}==ids
    assert analysis['all_outputs_bitwise_equal'] and analysis['original_event_values']==345600
    assert analysis['output_mse_vs_O5']==pytest.approx(dict(
        median=.005536172273439442,mean=.005053635851002639))
    for r in data:
        assert r['variant']=='o7' and r['paired_fp16']=='o5'
        assert r['guard']['ctas']==r['guard']['integer_ctas']==2048
        assert r['guard']['fallback_ctas']==r['guard']['invalid_ctas']==0
        assert r['mse_vs_v67']==r['max_abs_vs_v67']==0
        for key,value in timing_contract(r['mode'],100).items():assert r[key]==value
    old=rows(ROOT/'docs/evidence/a100_o378_roof_v99/runs/o378_roof_v99_o78/source_provenance.jsonl')
    old={r['sample_id']:r for r in old if r['variant']=='o7'}
    source=rows(RUN/'source_provenance.jsonl')
    assert len(source)==24 and all(r==old[r['sample_id']] for r in source)


def test_scope_exact_GPU_word_checks_and_same_actual_GEMM():
    env=read(RUN/'environment.json');validation=read(RUN/'validation.json')
    assert validation['passed'] and validation['count']==32 and validation['edge_count']==9
    assert validation['packed_word_exhaustive']['passed']
    assert validation['packed_word_exhaustive']['words']==131072
    assert env['args']['samples']==24 and env['args']['rounds']==3
    assert env['args']['warmup']==1000 and env['args']['repeats']==200 and env['args']['inner']==100
    assert env['source_quantization']=='original_FP16_direct_source_quantization_excluded'
    assert env['no_filtering'] and env['timing_scope']=='all_four_modes'
    assert not env['production_default_changed']
    for policy in ('0','1'):
        resource=env['resources'][policy]
        assert resource['kernel_symbol']=='adangel_roof_o78_eight_chain_candidate'
        assert resource['registers_per_thread']==168 and resource['local_size_bytes']==0
        assert resource['active_blocks_per_sm']==3 and resource['threads']==128
        assert resource['cta_tile']==[64,128,128] and resource['pipeline_stages']==2
        assert not resource['GEMM_modified']
    source=(ROOT/'scripts/benchmark_o7_nv4_swar.py').read_text()
    assert 'self.handles[0]=self.handles[1]' in source
    assert 'fn(self.handles[1],variant,1,0,sa,sw,state' in source
    for tool in ('memcheck','synccheck'):
        text=(E/('reports/o378_roof_v118_'+tool+'.log')).read_text()
        assert 'NVFP4 PACKED SWAR VALIDATION PASSED' in text
        assert 'ERROR SUMMARY: 0 errors' in text
        assert read(E/('runs/o378_v118_'+tool+'/validation.json'))==validation

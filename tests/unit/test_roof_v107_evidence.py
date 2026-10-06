"""v107 frozen compiler-only evidence; not a GPU performance/MSE claim."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_residency_axis_codegen import CONTROL,SYMBOL,STEM,generated_header,worth_runtime
from probe_roof_fullk_integer_codegen import static_entries

REPORTS=ROOT/'docs/evidence/a100_o378_roof_v107/reports'
DIRECTORY=REPORTS/'o378_roof_v107_codegen_r4'


def receipt():
    return json.loads((DIRECTORY/'codegen.json').read_text())


def test_exact_sources_artifacts_and_mapping_not_runtime():
    r=receipt()
    assert r['source_commit']=='e44142d9a4feddc247385c71a2cf8d8817a4fd80'
    for path,digest in r['sources'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for name,digest in r['artifact_sha256'].items():
        if name=='validate_coordinates' or Path(name).suffix=='.cubin':continue
        assert hashlib.sha256((DIRECTORY/name).read_bytes()).hexdigest()==digest,name
    assert (DIRECTORY/(STEM+'_generated.cuh')).read_text()==generated_header()
    assert r['coordinates']==dict(passed=True,threads=128,outputs=8192,
        operand_coordinates_compared=49152,gpu_execution=False)
    assert not r['changed_semantics'] and not r['production_default_changed']
    assert r['cta_tile']==[64,128,128] and r['threads']==128 and r['stages']==2
    assert r['partial_registers']==32 and r['independent_chains']==8
    assert r['logical_A_registers']+r['logical_B_registers']==48


def test_same_entry_native_INT4_and_exact_old_control_encoding():
    r=receipt();sass=(DIRECTORY/(STEM+'.sass')).read_text()
    ptx=(DIRECTORY/(STEM+'.ptx')).read_text()
    actual=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    assert actual==r['entries']
    for symbol in (CONTROL,SYMBOL):
        entry=actual[symbol]
        assert entry['native_s4_s4'] and entry['native_u4_s4']
        assert not entry['int8_mma'] and entry['all_copies_bypass_l1']
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
            if b.startswith('.visible .entry '+symbol+'('))
        assert all(x in body for x in ('cp.async.cg.shared.global','.s32.s4.s4.s32','.s32.u4.s4.s32'))
    old=ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    assert compare(old.read_text(),sass,'^'+CONTROL+'$')==r['control_comparison']
    assert r['control_comparison']['passed']


def test_real_compiled_work_does_not_pass_predeclared_potential_gate():
    r=receipt();live_text=(DIRECTORY/'liveness.txt').read_text()
    for symbol in (CONTROL,SYMBOL):assert analyze(live_text,symbol)==r['liveness'][symbol]
    old=r['liveness'][CONTROL];new=r['liveness'][SYMBOL]
    assert old['allocated_gpr']==new['allocated_gpr']==168
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    assert (a['static_instructions'],b['static_instructions'])==(383,379)
    assert a['max_live_gpr']==b['max_live_gpr']==166
    for loop in (a,b):
        ops=loop['opcode_counts']
        assert ops['IMMA.16864.S4.S4']==ops['IMMA.16864.U4.S4']==32
        assert ops['LDSM.16.M88.4']==16 and ops['LDGSTS.E.BYPASS.128']==10
        assert not any(v for op,v in ops.items() if op.startswith(('LDL','STL')))
    assert not worth_runtime(new,old) and not r['worth_runtime_validation']
    fallback=next(x for x in new['loops'] if x['kind']=='fp32_fallback')
    assert fallback['static_instructions']==453
    build=(DIRECTORY/'build.log').read_text()
    begin=build.index("Compiling entry function '"+SYMBOL+"'")
    end=build.index("Compiling entry function '"+CONTROL+"'",begin)
    assert '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' in build[begin:end]


def test_implementation_failures_are_preserved_not_optimization_iterations():
    assert 'auto full_a=thr.partition_A(ia),full_b=thr.partition_B(ib)' in (
        REPORTS/'o378_roof_v107_codegen/coordinates_build.log').read_text()
    assert "Assertion `cute::size<1>(sa)==1 && cute::size<2>(sa)==1' failed" in (
        REPORTS/'o378_roof_v107_codegen_r2/coordinates.log').read_text()
    assert 'auto atr=slice_mma.get_slice(threadIdx.x),aht=' in (
        REPORTS/'o378_roof_v107_codegen_r3/build.log').read_text()
    assert json.loads((DIRECTORY/'coordinates.log').read_text())['passed']

#!/usr/bin/env python3
"""v124 O3 W*A^T MMA: same logical tile, exact guard and original buffers.

Different from v107 (residency axis inside unchanged A*W^T), v47 warp aspect,
v45 fixed dimensions, v88 atom order, or v85 physically repacked operands.
No performance launch here. Preserve old generated files and encoded control.
Predeclared gate: >=3% fewer hot instructions OR >=16 fewer peak live GPR,
without more allocated registers, hot local, MMA/LDSM/copy/barrier work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import checked as checked_grouped
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o3_grouped_cta_candidate'
SYMBOL='adangel_roof_o3_transposed_mma_candidate'
STEM='o3_transposed_mma'
LIMITS=dict(min_hot_instruction_reduction=.03,min_live_gpr_reduction=16,
            max_allocated_gpr=168,max_hot_local=0,mma_each=32,ldsm=16,copies=9,barriers=1)


def cost_gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    count=lambda p:sum(v for op,v in b['opcode_counts'].items() if op.startswith(p))
    reduction=1-b['static_instructions']/a['static_instructions']
    live_saving=a['max_live_gpr']-b['max_live_gpr']
    checks=dict(meaningful_work_or_live_reduction=(reduction>=LIMITS['min_hot_instruction_reduction'] or
                live_saving>=LIMITS['min_live_gpr_reduction']),
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        hot_local=count('LDL')+count('STL')==LIMITS['max_hot_local'],
        native_math=count('IMMA.16864.S4.S4')==count('IMMA.16864.S4.U4')==LIMITS['mma_each'],
        same_supply=count('LDSM.')==LIMITS['ldsm'] and count('LDGSTS')==LIMITS['copies'],
        same_barrier=count('BAR')==LIMITS['barriers'])
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],
        static_reduction=reduction,live_gpr_saving=live_saving,
        scope='predeclared_compile_investment_gate_not_measured_speedup')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v89_o3_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();baseline=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    prior=checked_grouped(baseline,'o3');sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_bytes(f.read_bytes())
    sources=set(prior['sources'])|{
        'csrc/sm80/o3_transposed_mma_probe.cuh','csrc/sm80/roof_o3_transposed_mma_probe.cu',
        'tests/cuda/validate_transposed_mma_coordinates.cu','scripts/probe_transposed_mma_codegen.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O3_operand_transpose_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,
        nvcc=version,cutlass_commit=commit,production_default_changed=False,changed_semantics=False,
        logical_Y_tile=[64,128,128],internal_MMA_tile=[128,64,128],threads=128,stages=3,
        shared_bytes=50688,partial_registers=32,independent_chains=8,
        input_or_output_transpose_buffer=False,conversion_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        (out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coord=out/'validate_coordinates'
    run(flags+['-O2',str(ROOT/'tests/cuda/validate_transposed_mma_coordinates.cu'),'-o',str(coord)],'coordinates_build.log')
    run([str(coord)],'coordinates.log');r['coordinates']=json.loads((out/'coordinates.log').read_text())
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:
        raise ValueError('host CuTe mapping proof failed')
    src=str(ROOT/'csrc/sm80/roof_o3_transposed_mma_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    blocks=re.split(r'(?=Function\s*:\s*)',sass)
    candidate=next(b for b in blocks if b.startswith('Function') and b.splitlines()[0].split(':',1)[1].strip()==SYMBOL)
    es[SYMBOL]['native_s4_u4']=bool(re.search(r'\bIMMA[^;]*\.S4\.U4',candidate))
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s,formats in ((CONTROL,('.s32.u4.s4.s32','.s32.s4.s4.s32')),
                      (SYMBOL,('.s32.s4.u4.s32','.s32.s4.s4.s32'))):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global',)+formats):
            raise ValueError('same-entry PTX native math/supply missing')
    control=compare((baseline/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=control['passed']
    gate['passed'] &= control['passed']
    r.update(entries=es,liveness=live,cost_gate=gate,control_comparison=control,cubin_sha256=sha(cubin),
        artifact_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not es[SYMBOL]['native_s4_u4'] or not all(
            e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native INT4/control audit failed')
    print(json.dumps(dict(cost_gate=gate,coordinates=r['coordinates'],liveness=live),indent=2),flush=True)
    print('PASSED: numerical/resource/full24 validation next' if gate['passed'] else
          'FAILED: no candidate GPU execution or adjacent layout scan',flush=True)


if __name__=='__main__':main()

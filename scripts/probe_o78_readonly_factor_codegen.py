#!/usr/bin/env python3
"""v121: factor-only read-only supply, with unchanged shared payload pipeline.

Distinct from v43 register scale hoisting and v112 global payload fragments.
Only two metadata cp.async sites and their shared reads are replaced. Keep
the old shared reservation to avoid mixing this with an occupancy change.
Predeclare >=3% static loop reduction, <=168 registers, zero hot local and
unchanged native INT4/supply/barrier before investing in full24 GPU tests.
Static instruction counts are a potential gate, NOT measured speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_readonly_factor_candidate'
COPY_OLD='''  // Factors share the old scale-panel layout, commit/wait and CTA barrier.
  const unsigned first=threadIdx.x*4;
  if(threadIdx.x<16) copy16(s.activation_factors[slot]+first,
      af+group*m+blockIdx.y*64+first);
  if(threadIdx.x<32) copy16(s.weight_factors[slot]+first,
      wf+group*n+blockIdx.x*128+first);
'''
COPY_NEW='''  // v121: only matrix payload enters the original shared pipeline.
  // Af/Wf are immutable for this GEMM, prepared on the same stream earlier.
  // Reserve the original factor panels, but never read/write them here.
'''
LOAD_OLD='''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];'''
LOAD_NEW='''            const int coefficient=
                __ldg(af+size_t(group)*m+blockIdx.y*64+cute::get<0>(coord))*
                __ldg(wf+size_t(group)*n+blockIdx.x*128+cute::get<1>(coord));'''
LIMITS=dict(max_allocated_gpr=168,max_hot_local=0,min_static_reduction=.03,
    required_mma_each=32,required_ldsm=16,required_async_copy=8,
    required_barriers=1,minimum_active_ctas=3)


def generated_header(source):
    body=eight_header(source)
    if body.count(COPY_OLD)!=1 or body.count(LOAD_OLD)!=1:
        raise ValueError('v78 factor-copy/read boundary drift')
    return body.replace(COPY_OLD,COPY_NEW).replace(LOAD_OLD,LOAD_NEW).replace(
        'o78_eight_chain_experiment','o78_readonly_factor_experiment')


def cost_gate(old,new):
    a=next(l for l in old['loops'] if l['kind']=='integer')
    b=next(l for l in new['loops'] if l['kind']=='integer')
    count=lambda l,p:sum(n for op,n in l['opcode_counts'].items() if op.startswith(p))
    reduction=1-b['static_instructions']/a['static_instructions']
    checks=dict(allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        no_hot_local=count(b,'LDL')+count(b,'STL')==0,
        meaningful_work_reduction=reduction>=LIMITS['min_static_reduction'],
        native_math=count(b,'IMMA.16864.S4.S4')==32 and count(b,'IMMA.16864.U4.S4')==32,
        same_payload=count(b,'LDSM.')==16 and count(b,'LDGSTS')==8,
        same_barrier=count(b,'BAR')==LIMITS['required_barriers'],
        readonly_global_loads_present=count(b,'LDG')>0,
        no_shared_factor_reads=count(b,'LDS.')==0)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],
        static_reduction=reduction,old_allocated_gpr=old['allocated_gpr'],
        new_allocated_gpr=new['allocated_gpr'],
        scope='predeclared_compile_potential_gate_not_performance_acceptance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((a.baseline/'codegen.json').read_text())
    for path,value in prior['sources'].items():
        if sha(ROOT/path)!=value:raise ValueError('v78 source drift: '+path)
    if sha(a.baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header=out/'o78_readonly_factor_generated.cuh';header.write_text(generated_header(source))
    sources=set(prior['sources'])|{
        'csrc/sm80/roof_o78_readonly_factor_probe.cu','scripts/probe_o78_readonly_factor_codegen.py',
        'scripts/inspect_eight_chain_schedule.py','scripts/compare_a100_codegen.py'}
    receipt=dict(scope='O7_O8_factor_only_readonly_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},generated_header_sha256=sha(header),
        baseline_cubin_sha256=prior['cubin_sha256'],nvcc=version,cutlass_commit=commit,
        production_default_changed=False,cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        changed_semantics=False,supported_variants=['o7','o8'],limits=LIMITS,commands=[],
        memory_contract='Af_Wf_immutable_nonalias_output_same_stream_producer_before_GEMM',
        scale_guard_and_fallback='unchanged_v78_fullK_INT32_guard_else_original_FP32_path')
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_readonly_factor_probe.cu');cubin=out/'o78_readonly_factor.cubin'
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/'o78_readonly_factor.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_readonly_factor.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_readonly_factor.sass').read_text();ptx=(out/'o78_readonly_factor.ptx').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
            if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy audit failed')
        if s==SYMBOL and not re.search(r'\bld\.global(?:\.[a-z0-9]+)*\.nc\.',body):
            raise ValueError('read-only global factor path missing from PTX')
    schedules={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    gate=cost_gate(live[CONTROL],live[SYMBOL])
    control=compare((a.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    receipt.update(entries=entries,liveness=live,schedules=schedules,cost_gate=gate,
        cubin_sha256=sha(cubin),control_comparison=control,
        artifact_sha256={s:sha(out/s) for s in ('build.log','o78_readonly_factor.sass',
            'o78_readonly_factor.ptx','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('same-entry INT4/control audit failed')
    print(json.dumps(gate,indent=2),flush=True)
    print('Gate '+('PASSED: verify immutable/nonalias factors and full24 GPU tests next.' if gate['passed'] else
        'FAILED: no candidate launch, cache-policy neighbors or O3 migration.'),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""v119: pure-register native INT4 asm dependencies; one compile gate only.

Copy two operations from the pinned CuTe header, removing ONLY asm volatile.
The operation/constraints/register arrays and CuTe thread/value traits remain
exactly original. Never change third_party, a barrier, LDSM, scale or default.
This is not a claim that a warp collective is generally a pure scalar op:
the candidate's fixed uniform loop and complete warp participation still
require same-entry audit and, if the gate passes, GPU safety/output testing.
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
SYMBOL='adangel_roof_o78_register_mma_candidate'
OPERATIONS={
    'SM80_16x8x64_S32U4S4S32_TN':'AdangelRegisterOnlyU4S4',
    'SM80_16x8x64_S32S4S4S32_TN':'AdangelRegisterOnlyS4S4',
}
LIMITS=dict(max_allocated_gpr=168,max_hot_local=0,max_work_ratio=1.02,
    min_static_work_reduction=.05,alternative_max_allocated_gpr=128,
    required_mma_each=32,required_ldsm=16,required_async_copy=10)
REFERENCE='https://docs.nvidia.com/cuda/archive/12.8.0/inline-ptx-assembly/index.html#incorrect-optimization'


def original_operations(arch_text):
    result={}
    for name in OPERATIONS:
        match=re.search(r'^struct '+name+r'\n\{.*?^\};\n',arch_text,re.M|re.S)
        if match is None:
            raise ValueError('pinned CuTe operation boundary drift: '+name)
        body=match[0]
        if body.count('asm volatile(')!=1 or '"memory"' in body:
            raise ValueError('expected one pure-register asm and no hidden memory')
        result[name]=body
    return result


def generated_atoms(arch_text):
    old=original_operations(arch_text)
    # Keep the complete upstream BSD notice in future generated copies.
    notice=arch_text[:arch_text.index('#pragma once')]
    if 'SPDX-License-Identifier: BSD-3-Clause' not in notice:
        raise ValueError('pinned CuTe copyright/license notice missing')
    payload=[notice,'// Generated from pinned CuTe operations; only asm volatility changes.\n',
        '#pragma once\n#include <cute/atom/mma_traits_sm80.hpp>\n']
    for name,new in OPERATIONS.items():
        payload.append(old[name].replace(name,new).replace('asm volatile(','asm('))
    payload.append('namespace cute {\n')
    for name,new in OPERATIONS.items():
        payload.append('template<> struct MMA_Traits<::'+new+'> : MMA_Traits<'+name+'> {};\n')
    payload.append('} // namespace cute\n')
    return ''.join(payload)


def generated_body(source):
    text=eight_header(source)
    for old,new in OPERATIONS.items():
        marker='cute::MMA_Atom<cute::'+old+'>'
        if text.count(marker)!=1:
            raise ValueError('v78 atom boundary drift: '+old)
        text=text.replace(marker,'cute::MMA_Atom<::'+new+'>')
    return text.replace('o78_eight_chain_experiment','o78_register_mma_experiment')


def gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    count=lambda prefix:sum(n for op,n in b['opcode_counts'].items() if op.startswith(prefix))
    ratio=b['static_instructions']/a['static_instructions']
    work_gain=1-ratio
    checks=dict(
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        no_hot_local=count('LDL')+count('STL')==0,
        bounded_total_work=ratio<=LIMITS['max_work_ratio'],
        meaningful_potential=work_gain>=LIMITS['min_static_work_reduction'] or
            new['allocated_gpr']<=LIMITS['alternative_max_allocated_gpr'],
        native_math=count('IMMA.16864.S4.S4')==32 and count('IMMA.16864.U4.S4')==32,
        unchanged_supply=count('LDSM.')==16 and count('LDGSTS')==10,
        unchanged_barrier=count('BAR')==sum(n for op,n in a['opcode_counts'].items()
            if op.startswith('BAR')),
    )
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],
        static_work_ratio=ratio,static_work_reduction=work_gain,
        old_allocated_gpr=old['allocated_gpr'],new_allocated_gpr=new['allocated_gpr'],
        scope='compile_potential_only_not_latency_concurrency_or_GPU_safety')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();baseline=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not baseline.is_relative_to(ROOT):
        parser.error('fresh output and existing baseline must be inside repository')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for name,value in prior['sources'].items():
        if sha(ROOT/name)!=value:raise ValueError('v78 source drift: '+name)
    if sha(baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA12.8/CUTLASS required')
    arch=cutlass/'include/cute/arch/mma_sm80.hpp'
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    control=eight_header(source)
    if control!=(baseline/'o78_eight_chain_generated.cuh').read_text():
        raise ValueError('baseline generated body drift')
    out.mkdir(parents=True)
    headers={
        'o78_eight_chain_generated.cuh':control,
        'register_mma_generated.cuh':generated_atoms(arch.read_text()),
        'o78_register_mma_generated.cuh':generated_body(source),
    }
    for name,body in headers.items():(out/name).write_text(body)
    sources=set(prior['sources'])|{
        'scripts/probe_register_mma_codegen.py','scripts/compare_a100_codegen.py',
        'scripts/inspect_eight_chain_schedule.py','csrc/sm80/roof_o78_register_mma_probe.cu'}
    receipt=dict(scope='O7_O8_register_only_MMA_compile_gate_not_runtime_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},
        generated_sha256={s:sha(out/s) for s in headers},
        upstream_arch_sha256=sha(arch),baseline_cubin_sha256=prior['cubin_sha256'],
        reference=REFERENCE,nvcc=version,cutlass_commit=commit,limits=LIMITS,
        production_default_changed=False,changed_math=False,changed_asm_volatility=True,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        supported_variants=['o7','o8'],commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo',
        '-arch=sm_80','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_register_mma_probe.cu')
    cubin=out/'o78_register_mma.cubin'
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/'o78_register_mma.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_register_mma.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_register_mma.sass').read_text();ptx=(out/'o78_register_mma.ptx').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
            if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy audit failed')
    control_check=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    cost=gate(live[CONTROL],live[SYMBOL])
    cost['control_encoding_unchanged']=control_check['passed']
    cost['passed'] &= control_check['passed']
    # Same symbol-independent encoded stream means the compiler hypothesis
    # did not change executable work at all. No timing experiment can claim
    # an improvement from this source change in that case.
    from compare_a100_codegen import instructions as encoded
    old_words=encoded(sass,'^'+CONTROL+'$')[CONTROL]
    new_words=encoded(sass,'^'+SYMBOL+'$')[SYMBOL]
    cost['full_entry_encoded_equal']=old_words==new_words
    schedule={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    receipt.update(cubin_sha256=sha(cubin),entries=entries,liveness=live,
        control_comparison=control_check,cost_gate=cost,schedule=schedule,
        artifact_sha256={s:sha(out/s) for s in ('build.log','ptx_build.log',
            'o78_register_mma.sass','o78_register_mma.ptx','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not control_check['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
        not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('native math/old control audit failed')
    print(json.dumps(cost,indent=2),flush=True)
    print('Gate '+('PASSED: full GPU validation required before reporting gains.' if cost['passed']
        else 'FAILED: no candidate launch, no neighboring volatility variants or O3 migration.'),flush=True)


if __name__=='__main__':main()

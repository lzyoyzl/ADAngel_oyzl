#!/usr/bin/env python3
"""One bounded coefficient handoff, on the frozen v78 eight-chain kernel.

After every low-K64_0 MMA has consumed B0, its eight register words are dead.
Reuse that fragment as an eight-coefficient window (two M atoms, one N atom).
Do not allocate a full output coefficient table or change the input layout.
This is an independent compile gate, not a production implementation switch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from analyze_o78_coefficient_codegen import dependencies
from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions, trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o78_recycled_coefficient_candidate'
STEM = 'o78_recycled_coefficient'
LAST_MMA = '''      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
      });
'''
WEIGHTING = '''      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
          });
        });
      });
'''
PREPARE = '''      // B0's final consumers were the preceding eight low-K64_0 MMAs.
      // The next N64 slice reloads B0 before it is used as an operand again.
      // recast is a bit view of the same register fragment, not a new array.
      auto coefficients=cute::recast<int32_t>(b0);
      static_assert(decltype(cute::size(coefficients))::value==8);
      auto prepare_coefficients=[&](auto ni) {
        const auto full_ni=nb*cute::_4{}+ni;
        o1_static_for<0,2>([&](auto mi) {
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int row=s.activation_factors[slot][cute::get<0>(coord)];
            const int column=s.weight_factors[slot][cute::get<1>(coord)];
            int product;
            // Keep the independent product; no volatile or memory fence.
            asm("mul.lo.s32 %0, %1, %2;" : "=r"(product) : "r"(row), "r"(column));
            coefficients(mi*cute::_4{}+vi)=product;
          });
        });
      };
      // Only eight coefficients, not32/64, overlap the final low MMA wave.
      prepare_coefficients(cute::_0{});
'''
CONSUME = '''      o1_static_for<0,4>([&](auto ni) {
        if constexpr(decltype(ni)::value!=0) prepare_coefficients(ni);
        const auto full_ni=nb*cute::_4{}+ni;
        o1_static_for<0,2>([&](auto mi) {
          o1_static_for<0,4>([&](auto vi) {
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficients(mi*cute::_4{}+vi);
          });
        });
      });
'''
LIMITS = dict(max_allocated_gpr=168, max_hot_local=0, max_static_ratio=1.0,
              ldsm=16, async_copy=10, native_mma_each=32, required_updates=64,
              min_independent_chains=8)


def generated_header(source):
    old = eight_header(source)
    if old.count(LAST_MMA) != 1 or old.count(WEIGHTING) != 1:
        raise ValueError('v78 final MMA/weighting source drift')
    if LAST_MMA+WEIGHTING not in old:
        raise ValueError('v78 weighting must immediately follow the last MMA wave')
    return old.replace(LAST_MMA+WEIGHTING, PREPARE+LAST_MMA+CONSUME).replace(
        'o78_eight_chain_experiment', 'o78_recycled_coefficient_experiment')


def dependency_audit(sass, symbol, live):
    schedule = trace(sass, symbol, live)
    ops = instructions(sass, symbol, schedule['loop'])
    result = dependencies([(pc, op.split()[0], op) for pc, op in ops])
    return dict(counts={k:len(v) for k,v in result.items()}, details=result,
                peak_started_not_finished_chains=schedule['peak_started_not_finished_chains'],
                scope='static def-use only, not hardware overlap or measured speedup')


def gate(old, new, dep):
    a = next(x for x in old['loops'] if x['kind']=='integer')
    b = next(x for x in new['loops'] if x['kind']=='integer')
    count = lambda prefix:sum(v for op,v in b['opcode_counts'].items() if op.startswith(prefix))
    old_lds = sum(v for op,v in a['opcode_counts'].items() if op=='LDS' or op.startswith('LDS.'))
    new_lds = sum(v for op,v in b['opcode_counts'].items() if op=='LDS' or op.startswith('LDS.'))
    ratio = b['static_instructions']/a['static_instructions']
    checks = dict(
        register_budget=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        no_hot_local=count('LDL')+count('STL')==0,
        no_static_work_growth=ratio<=LIMITS['max_static_ratio'],
        same_native_math=count('IMMA.16864.S4.S4')==count('IMMA.16864.U4.S4')==32,
        same_payload_supply=count('LDSM.')==16 and count('LDGSTS')==10,
        no_scalar_load_growth=new_lds<=old_lds,
        all_updates_coefficient_based=dep['counts']['coefficient_based_accumulator_updates']==64,
        all_coefficients_independent=dep['counts']['independent_coefficient_multiplies']==64,
        no_partial_first_multiply=dep['counts']['partial_first_multiplies']==0,
        eight_chains_preserved=dep['peak_started_not_finished_chains']>=8)
    return dict(passed=all(checks.values()), checks=checks, limits=LIMITS,
                static_ratio=ratio, old_static=a['static_instructions'],
                new_static=b['static_instructions'], old_scalar_lds=old_lds,
                new_scalar_lds=new_lds,
                scope='predeclared compile potential only; full24/MSE/safety still required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((args.baseline/'codegen.json').read_text())
    for path,value in prior['sources'].items():
        if sha(ROOT/path)!=value:raise ValueError('v78 source drift: '+path)
    if sha(args.baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin'); cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one production extension required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    header=eight_header(source)
    if header!=(args.baseline/'o78_eight_chain_generated.cuh').read_text():
        raise ValueError('frozen v78 header drift')
    (out/'o78_eight_chain_generated.cuh').write_text(header)
    (out/(STEM+'_generated.cuh')).write_text(generated_header(source))
    sources=set(prior['sources'])|{
        'scripts/probe_o78_recycled_coefficient_codegen.py',
        'scripts/analyze_o78_coefficient_codegen.py', 'scripts/inspect_eight_chain_schedule.py',
        'scripts/compare_a100_codegen.py', 'scripts/inspect_o78_register_liveness.py',
        'csrc/sm80/roof_o78_recycled_coefficient_probe.cu'}
    receipt=dict(scope='v141_bounded_coefficient_register_handoff_compile_only',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)}, commands=[],
        baseline_cubin_sha256=prior['cubin_sha256'], nvcc=version, cutlass_commit=commit,
        cta_tile=[64,128,128], threads=128, stages=2, shared_bytes=34304,
        coefficient_slots=8, limits=LIMITS, production_default_changed=False,
        conversion_changed=False, native_extension_rebuilt=False, candidate_gpu_launched=False,
        new_performance_result=False, new_MSE_measured=False,
        extension_sha256_before=sha(extensions[0]))
    def run(command,name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
           '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
           str(ROOT/'csrc/sm80/roof_o78_recycled_coefficient_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text(); live=(out/'liveness.txt').read_text()
    receipt['control_comparison']=compare((args.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    if not receipt['control_comparison']['passed']:raise ValueError('v78 control encoding changed')
    receipt['entries']=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    ptx=(out/(STEM+'.ptx')).read_text()
    for symbol,entry in receipt['entries'].items():
        if not(entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
               and entry['all_copies_bypass_l1']):raise ValueError('native math/copy audit failed')
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                  if b.startswith('.visible .entry '+symbol+'('))
        if not all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX audit failed')
    receipt['liveness']={s:analyze(live,s) for s in (CONTROL,SYMBOL)}
    receipt['dependency_audit']={s:dependency_audit(sass,s,receipt['liveness'][s]) for s in (CONTROL,SYMBOL)}
    receipt['compile_gate']=gate(receipt['liveness'][CONTROL],receipt['liveness'][SYMBOL],
                                 receipt['dependency_audit'][SYMBOL])
    receipt['extension_sha256_after']=sha(extensions[0])
    if receipt['extension_sha256_after']!=receipt['extension_sha256_before']:raise ValueError('extension changed')
    receipt['cubin_sha256']=sha(cubin)
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['compile_gate'],indent=2),flush=True)


if __name__=='__main__':main()

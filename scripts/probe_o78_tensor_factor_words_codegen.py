#!/usr/bin/env python3
"""v130: one exact U32-fragment repair to v129's diagnosed byte packing.

Same CuTe coordinates, rank-one U8 MMA, original INT4 payload and width guard.
No layout/chain/stage sweep or relaxed cost gate. No GPU launch here.
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
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_o78_tensor_factor_codegen import SETUP, OLD, LIMITS, cost_gate, analyze_candidate
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_tensor_factor_words_candidate'
STEM='o78_tensor_factor_words'
NEW='''      // Host CuTe proof: within each packed U32 only byte0 can have K=0.
      // Preserve the exact v129 byte image, without building uint8 fragments.
      o1_static_for<0,2>([&](auto mi) {
        const auto pa0=factor_ac(cute::_0{},mi,cute::_0{});
        const auto pa1=factor_ac(cute::_4{},mi,cute::_0{});
        const uint32_t fa0=cute::get<1>(pa0)==0?static_cast<uint32_t>(
            s.activation_factors[slot][cute::get<0>(pa0)]):0u;
        const uint32_t fa1=cute::get<1>(pa1)==0?static_cast<uint32_t>(
            s.activation_factors[slot][cute::get<0>(pa1)]):0u;
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          const auto pb=factor_bc(cute::_0{},full_ni,cute::_0{});
          const uint32_t fb=cute::get<1>(pb)==0?static_cast<uint32_t>(
              s.weight_factors[slot][cute::get<0>(pb)]):0u;
          auto coefficient=cute::make_tensor<uint32_t>(cute::make_shape(cute::_4{}));
          // Official CUTLASS packed-register API, not hand-authored lane rules.
          cute::SM80_16x8x16_S32U8U8S32_TN::fma(
              coefficient(0),coefficient(1),coefficient(2),coefficient(3),
              fa0,fa1,fb,0u,0u,0u,0u);
          o1_static_for<0,4>([&](auto vi) {
            // U8 outer products are [0,65025], so this cast is exact.
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*static_cast<int>(coefficient(vi));
          });
        });
      });
'''


def generated_header():
    old=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    marker='  constexpr int Groups=32;'
    if old.count(OLD)!=1 or old.count(marker)!=1:raise ValueError('v78 source boundary drift')
    return old.replace(marker,SETUP+marker).replace(OLD,NEW).replace(
        'o78_eight_chain_experiment','o78_tensor_factor_words_experiment')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();base=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    old=json.loads((base/'codegen.json').read_text())
    for name,digest in old['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v78 source drift: '+name)
    if sha(base/'o78_eight_chain.cubin')!=old['cubin_sha256']:raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    (out/'o78_eight_chain_generated.cuh').write_bytes((base/'o78_eight_chain_generated.cuh').read_bytes())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(old['sources'])|{'csrc/sm80/roof_o78_tensor_factor_words_probe.cu',
        'scripts/probe_o78_tensor_factor_words_codegen.py','scripts/probe_o78_tensor_factor_codegen.py',
        'tests/cuda/validate_tensor_factor_words.cu','tests/cuda/validate_tensor_factor_coordinates.cu',
        'scripts/inspect_o78_register_liveness.py','scripts/compare_a100_codegen.py',
        'scripts/probe_roof_fullk_integer_codegen.py'}
    r=dict(scope='O78_exact_packed_coefficient_register_repair_compile_gate',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,
        nvcc=version,cutlass_commit=commit,baseline_cubin_sha256=old['cubin_sha256'],
        production_default_changed=False,new_candidate_GPU_executed=False,conversion_changed=False,
        changed_semantics=False,cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        previous_failure_preserved='docs/evidence/a100_o378_roof_v129',
        single_repair_not_layout_parameter_sweep=True)
    def run(cmd,name):
        r['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    exe=out/'coordinate_check'
    run(flags+['-O2',str(ROOT/'tests/cuda/validate_tensor_factor_words.cu'),'-o',str(exe)],'coordinate_build.log')
    run([str(exe)],'coordinates.json');r['coordinates']=json.loads((out/'coordinates.json').read_text())
    assert r['coordinates']['passed'] and not r['coordinates']['gpu_execution']
    src=str(ROOT/'csrc/sm80/roof_o78_tensor_factor_words_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live_text=(out/'liveness.txt').read_text()
    live={CONTROL:analyze(live_text,CONTROL),SYMBOL:analyze_candidate(live_text,SYMBOL)}
    entry=next(x for x in re.split(r'(?=\.visible \.entry )',ptx) if x.startswith('.visible .entry '+SYMBOL+'('))
    assert all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32',
        '.s32.s4.s4.s32','mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32'))
    control=compare((base/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=control['passed']
    gate['passed'] &= control['passed']
    r.update(entries=entries,liveness=live,cost_gate=gate,control_comparison=control,
        cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('progress.json','codegen.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    assert control['passed'] and all(e['native_u4_s4'] and e['native_s4_s4'] and
        e['all_copies_bypass_l1'] for e in entries.values())
    print(json.dumps(dict(cost_gate=gate,liveness=live),indent=2),flush=True)
    print('PASSED: check real full24 coverage, resource, numeric and safety before timing' if gate['passed'] else
        'FAILED: stop this exact-word repair without GPU timing or further packing scans',flush=True)


if __name__=='__main__':main()

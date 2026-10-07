#!/usr/bin/env python3
"""v129: exact rank-one scale products via a U8/U8 Tensor Core helper.

Only replace the64 scalar coefficient products per thread/G128. The payload
still uses64 original native INT4 MMA, same G128 math/order and full-K guard.
An explicit on-GPU U8 range check chooses original integer body if too wide.
No narrowing/clipping, new quantizer, default switch or GPU timing here.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_tensor_factor_candidate'
STEM='o78_tensor_factor'
REFERENCE='https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-i8'
LIMITS=dict(max_allocated_gpr=168,max_hot_local=1,max_work_ratio=1.01,
    min_imad_reduction=.20,min_runtime_cta=3,min_opportunity_fraction=.50,
    payload_mma_each=32,coefficient_mma=16,ldsm=16,copies=10,barriers=1)
SETUP='''  using FactorAtom=cute::MMA_Atom<cute::SM80_16x8x16_S32U8U8S32_TN>;
  using FactorMma=cute::TiledMMA<FactorAtom,
      cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>>,
      cute::Tile<cute::_64,cute::_128,cute::_16>>;
  FactorMma factor_mma;
  auto ft=factor_mma.get_slice(threadIdx.x);
  auto factor_ac=ft.partition_A(cute::make_identity_tensor(
      cute::make_shape(cute::_64{},cute::_16{})));
  auto factor_bc=ft.partition_B(cute::make_identity_tensor(
      cute::make_shape(cute::_128{},cute::_16{})));
  // Host CuTe verifier checks the complete A/B shape and exact C-coordinate
  // equality with the unchanged payload MMA. No hand-written lane mapping.
'''
OLD='''      o1_static_for<0,2>([&](auto mi) {
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
NEW='''      // Only k=0 carries nonzero metadata; remaining15 products are zero.
      // The CTA guard proves U8 casts below are exact, including0 and255.
      o1_static_for<0,2>([&](auto mi) {
        auto fa=cute::make_tensor<uint8_t>(cute::make_shape(cute::_8{}));
        o1_static_for<0,8>([&](auto vi) {
          const auto p=factor_ac(vi,mi,cute::_0{});
          fa(vi)=cute::get<1>(p)==0?static_cast<uint8_t>(
              s.activation_factors[slot][cute::get<0>(p)]):uint8_t(0);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          auto fb=cute::make_tensor<uint8_t>(cute::make_shape(cute::_4{}));
          o1_static_for<0,4>([&](auto vi) {
            const auto p=factor_bc(vi,full_ni,cute::_0{});
            fb(vi)=cute::get<1>(p)==0?static_cast<uint8_t>(
                s.weight_factors[slot][cute::get<0>(p)]):uint8_t(0);
          });
          auto coefficient=cute::make_tensor<int>(cute::make_shape(cute::_4{}));
          cute::clear(coefficient);
          cute::gemm(FactorAtom{},coefficient,fa,fb,coefficient);
          o1_static_for<0,4>([&](auto vi) {
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient(vi);
          });
        });
      });
'''


def generated_header():
    source=eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    marker='  constexpr int Groups=32;'
    if source.count(OLD)!=1 or source.count(marker)!=1:
        raise ValueError('v78 coefficient/metadata source boundary drift')
    return source.replace(marker,SETUP+marker).replace(OLD,NEW).replace(
        'o78_eight_chain_experiment','o78_tensor_factor_experiment')


def analyze_candidate(text):
    """Exactly3 complete K loops: FP32, original INT32, metadata-TC INT32.

    The width guard is a separate short loop, not part of the hot-loop cost.
    Its global reads, predicate reduction and barrier are charged at runtime.
    """
    block=next(b for b in re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
        if re.match(r'//-+ \.text\.'+SYMBOL+r'\s',b))
    allocated=int(re.search(r'SHI_REGISTERS=(\d+)',block)[1])
    ops=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label:pending.append(label[1])
        ins=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if ins:
            pc=int(ins[1],16)
            for name in pending:labels[name]=pc
            pending.clear();ops.append((pc,ins[2].strip(),int(ins[3])))
    loops=[]
    for pc,inst,live in ops:
        branch=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',inst)
        if not branch or labels[branch[1]]>=pc:continue
        region=[x for x in ops if labels[branch[1]]<=x[0]<=pc]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',x[1]).split()[0] for x in region)
        if sum(v for op,v in counts.items() if op.startswith('IMMA.16864.'))!=64:continue
        helper=counts['IMMA.16816.U8.U8']
        kind=('tensor_factor_integer' if helper==16 else
              'fp32_fallback' if counts.get('I2F',0) else 'integer_fallback')
        if helper not in (0,16):raise ValueError('unexpected metadata MMA work')
        loops.append(dict(kind=kind,begin_pc=hex(region[0][0]),end_pc=hex(region[-1][0]),
            static_instructions=len(region),max_live_gpr=max(x[2] for x in region),
            opcode_counts=dict(sorted(counts.items()))))
    if len(loops)!=3 or {x['kind'] for x in loops}!={'tensor_factor_integer','fp32_fallback','integer_fallback'}:
        raise ValueError('expected exact helper/native integer/FP32 loop set: '+str(loops))
    return dict(symbol=SYMBOL,allocated_gpr=allocated,loops=loops,
        function_max_live_gpr=max(x[2] for x in ops),cta_threads=128,
        interpretation='static_binary_cost_not_dynamic_work_or_runtime_safety')


def cost_gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='tensor_factor_integer')
    count=lambda x,p:sum(v for op,v in x['opcode_counts'].items() if op.startswith(p))
    ratio=b['static_instructions']/a['static_instructions']
    reduction=1-count(b,'IMAD')/count(a,'IMAD')
    checks=dict(allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        hot_local=count(b,'LDL')+count(b,'STL')<=LIMITS['max_hot_local'],
        work=ratio<=LIMITS['max_work_ratio'],integer_service=reduction>=LIMITS['min_imad_reduction'],
        payload_int4=count(b,'IMMA.16864.S4.S4')==count(b,'IMMA.16864.U4.S4')==32,
        metadata_int8=count(b,'IMMA.16816.U8.U8')==16,
        supply=count(b,'LDSM.')==16 and count(b,'LDGSTS')==10,
        hot_barrier=count(b,'BAR')==1)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],work_ratio=ratio,
        old_imad=count(a,'IMAD'),new_imad=count(b,'IMAD'),imad_reduction=reduction,
        extra_INT8_scope='scale_outer_product_only_not_payload',
        scope='predeclared_compile_gate_excludes_online_range_guard_not_performance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();baseline=args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v78 source drift: '+name)
    if sha(baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    (out/'o78_eight_chain_generated.cuh').write_bytes((baseline/'o78_eight_chain_generated.cuh').read_bytes())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'csrc/sm80/roof_o78_tensor_factor_probe.cu',
        'scripts/probe_o78_tensor_factor_codegen.py','tests/cuda/validate_tensor_factor_coordinates.cu',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O7_O8_tensor_scale_outer_product_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,reference=REFERENCE,
        nvcc=version,cutlass_commit=commit,production_default_changed=False,changed_semantics=False,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,partial_registers=32,
        conversion_changed=False,new_candidate_GPU_executed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coord=out/'coordinate_check'
    run(flags+['-O2',str(ROOT/'tests/cuda/validate_tensor_factor_coordinates.cu'),'-o',str(coord)],'coordinate_build.log')
    run([str(coord)],'coordinates.json');r['coordinates']=json.loads((out/'coordinates.json').read_text())
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:raise ValueError('host CuTe coordinate gate failed')
    src=str(ROOT/'csrc/sm80/roof_o78_tensor_factor_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    ltext=(out/'liveness.txt').read_text()
    live={CONTROL:analyze(ltext,CONTROL),SYMBOL:analyze_candidate(ltext)}
    body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+SYMBOL+'('))
    if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32',
        '.s32.s4.s4.s32','mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32')):
        raise ValueError('same-entry PTX payload/metadata math missing')
    control=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate=cost_gate(live[CONTROL],live[SYMBOL]);gate['control_encoding_unchanged']=control['passed']
    gate['passed'] &= control['passed']
    r.update(entries=es,liveness=live,cost_gate=gate,control_comparison=control,
        cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f) for f in out.iterdir()
            if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
        e['all_copies_bypass_l1'] for e in es.values()):raise ValueError('payload/control audit failed')
    print(json.dumps(dict(cost_gate=gate,liveness=live),indent=2),flush=True)
    print('PASSED: full24 data coverage/resource/correctness before performance' if gate['passed'] else
          'FAILED: stop without candidate GPU execution or neighboring layout scan',flush=True)


if __name__=='__main__':main()

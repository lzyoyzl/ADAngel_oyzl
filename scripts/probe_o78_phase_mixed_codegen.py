#!/usr/bin/env python3
"""v120: fixed 1/3 four-chain + 2/3 eight-chain CTA scheduling compile gate.

Not a uniform chain-count sweep. Only scheduling varies across CTAs, outside
the K loop. No assumption that three adjacent CTAs reside on the same SM.
Phase overlap is a latency hypothesis, not a static instruction-count gain.
Stop a failed gate; no ratio/seed neighbors or O3 migration without evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header, MERGED
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_phase_mixed_candidate'
LIMITS=dict(max_allocated_gpr=168,max_hot_local=0,max_weighted_work_ratio=1.03,
    required_mma_each=32,required_ldsm=16,required_async_copy=10,
    four_chain_fraction=[1,3],minimum_active_ctas=3)
FOUR='''      // Four merged N chains for one M atom at a time. Same exact dot.
      o1_static_for<0,2>([&](auto mi) {
        auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        cute::clear(partial);
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,h0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(HA{},p,h1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,16>([&](auto i) { partial(i)*=16; });
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,a0(cute::_,mi,cute::_0{}),b0(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,ni);
          cute::gemm(LA{},p,a1(cute::_,mi,cute::_0{}),b1(cute::_,ni,cute::_0{}),p);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,ni)*coefficient;
          });
        });
      });
'''


def generated_four(source):
    original=eight_header(source)
    if original.count(MERGED)!=1:
        raise ValueError('v78 merged body drift')
    return original.replace(MERGED,FOUR).replace(
        'o78_eight_chain_experiment','o78_four_chain_experiment')


def mixed_liveness(text,symbol=SYMBOL):
    """Same natural-loop rule as v75, but require TWO integer paths.

    No alteration of the existing one-integer-loop evidence parser. Counts
    are per programmed G128 path, never summed as if both paths execute.
    """
    sections=re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
    block=next((b for b in sections if re.match(
        r'//-+ \.text\.'+re.escape(symbol)+r'\s',b)),None)
    if block is None:raise ValueError('exact mixed entry missing')
    allocated=re.search(r'SHI_REGISTERS=(\d+)',block)
    if not allocated:raise ValueError('allocated register metadata missing')
    instructions=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label:pending.append(label[1])
        ins=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if ins:
            pc=int(ins[1],16)
            for name in pending:labels[name]=pc
            pending.clear()
            instructions.append(dict(pc=pc,instruction=ins[2].strip(),live_gpr=int(ins[3])))
    if not instructions:raise ValueError('no count-mode instructions')
    loops=[]
    for ins in instructions:
        branch=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',ins['instruction'])
        if branch and branch[1] not in labels:raise ValueError('unresolved branch target')
        if not branch or labels[branch[1]]>=ins['pc']:continue
        region=[i for i in instructions if labels[branch[1]]<=i['pc']<=ins['pc']]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',i['instruction']).split()[0] for i in region)
        if sum(n for op,n in counts.items() if op.startswith('IMMA.'))!=64:continue
        peak=max(i['live_gpr'] for i in region)
        loops.append(dict(kind='fp32_fallback' if counts.get('I2F',0) else 'integer',
            begin_pc=hex(region[0]['pc']),end_pc=hex(region[-1]['pc']),
            static_instructions=len(region),max_live_gpr=peak,
            opcode_counts=dict(sorted(counts.items())),
            peak_pcs=[hex(i['pc']) for i in region if i['live_gpr']==peak]))
    if len(loops)!=3 or Counter(l['kind'] for l in loops)!={'integer':2,'fp32_fallback':1}:
        raise ValueError('require exactly two integer paths and original fallback')
    return dict(symbol=symbol,allocated_gpr=int(allocated[1]),
        function_max_live_gpr=max(i['live_gpr'] for i in instructions),loops=loops,
        interpretation='static_paths_not_dynamic_concurrency_or_latency')


def schedules(sass,live):
    result=[]
    for loop in live['loops']:
        if loop['kind']=='integer':
            result.append(trace(sass,SYMBOL,dict(loops=[loop])))
    return result


def cost_gate(old,new,paths):
    control=next(l for l in old['loops'] if l['kind']=='integer')
    by_chains={p['peak_started_not_finished_chains']:p['loop'] for p in paths}
    count=lambda l,p:sum(n for op,n in l['opcode_counts'].items() if op.startswith(p))
    work=(by_chains[4]['static_instructions']+2*by_chains[8]['static_instructions'])/3 if set(by_chains)=={4,8} else None
    checks=dict(distinct_four_and_eight_paths=len(paths)==2 and set(by_chains)=={4,8},
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        no_hot_local=all(count(p['loop'],'LDL')+count(p['loop'],'STL')==0 for p in paths),
        same_math=all(count(p['loop'],'IMMA.16864.S4.S4')==32 and
            count(p['loop'],'IMMA.16864.U4.S4')==32 for p in paths),
        same_supply=all(count(p['loop'],'LDSM.')==16 and count(p['loop'],'LDGSTS')==10 for p in paths),
        same_barrier=all(count(p['loop'],'BAR')==count(control,'BAR') for p in paths),
        bounded_weighted_work=work is not None and work/control['static_instructions']<=LIMITS['max_weighted_work_ratio'])
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        baseline_static=control['static_instructions'],weighted_static=work,
        weighted_work_ratio=None if work is None else work/control['static_instructions'],
        paths={str(c):dict(begin_pc=l['begin_pc'],static_instructions=l['static_instructions'])
            for c,l in sorted(by_chains.items())},
        scope='phase_overlap_hypothesis_gate_not_speedup_or_proof_of_same_SM_placement')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();baseline=a.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT) or not baseline.is_relative_to(ROOT):
        parser.error('fresh output and existing baseline must be inside repository')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for path,digest in prior['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('v78 source drift: '+path)
    if sha(baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:
        raise ValueError('baseline cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA12.8/CUTLASS required')
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    if eight_header(source)!=(baseline/'o78_eight_chain_generated.cuh').read_text():
        raise ValueError('generated control drift')
    out.mkdir(parents=True)
    headers={'o78_eight_chain_generated.cuh':eight_header(source),
             'o78_four_chain_generated.cuh':generated_four(source)}
    for name,body in headers.items():(out/name).write_text(body)
    sources=set(prior['sources'])|{'csrc/sm80/roof_o78_phase_mixed_probe.cu',
        'scripts/probe_o78_phase_mixed_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/compare_a100_codegen.py'}
    receipt=dict(scope='O7_O8_fixed_CTA_phase_mix_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},
        generated_sha256={s:sha(out/s) for s in headers},
        baseline_cubin_sha256=prior['cubin_sha256'],nvcc=version,cutlass_commit=commit,
        production_default_changed=False,cta_tile=[64,128,128],threads=128,stages=2,
        shared_bytes=34304,changed_math=False,changed_pipeline=False,
        supported_variants=['o7','o8'],limits=LIMITS,commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo',
        '-arch=sm_80','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_phase_mixed_probe.cu');cubin=out/'o78_phase_mixed.cubin'
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/'o78_phase_mixed.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_phase_mixed.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_phase_mixed.sass').read_text();ptx=(out/'o78_phase_mixed.ptx').read_text()
    entries=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={CONTROL:analyze((out/'liveness.txt').read_text(),CONTROL),
          SYMBOL:mixed_liveness((out/'liveness.txt').read_text())}
    paths=schedules(sass,live[SYMBOL])
    gate=cost_gate(live[CONTROL],live[SYMBOL],paths)
    control=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    for symbol in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
            if b.startswith('.visible .entry '+symbol+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native INT4/copy PTX audit failed')
    receipt.update(cubin_sha256=sha(cubin),entries=entries,liveness=live,schedules=paths,
        cost_gate=gate,control_comparison=control,
        artifact_sha256={s:sha(out/s) for s in ('build.log','ptx_build.log',
            'o78_phase_mixed.sass','o78_phase_mixed.ptx','resources.txt','liveness.txt')})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
        not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('native math/original control audit failed')
    print(json.dumps(gate,indent=2),flush=True)
    print('Gate '+('PASSED: full GPU validation still required.' if gate['passed'] else
        'FAILED: no candidate launch, no ratio/seed neighbors or O3 migration.'),flush=True)


if __name__=='__main__':main()

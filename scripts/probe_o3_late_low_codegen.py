#!/usr/bin/env python3
"""v131: defer low-A LDSM within the SAME group, with unchanged register reuse.

v89 already delays two of four low-A loads. Test only the remaining startup
loads: after high/high MMA but before high*16, then reuse for both N64 slices.
Unlike v128 there is no next-stage transition, cached factors or extra state.
This script compiles/audits only; a passed gate is NOT a performance result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions, trace
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import checked as checked_grouped, generated_headers
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o3_grouped_cta_candidate'
SYMBOL='adangel_roof_o3_late_low_candidate'
STEM='o3_late_low'
LIMITS=dict(max_work_ratio=1.02,max_allocated_gpr=168,max_hot_local=0,
    min_actual_cta=3,min_first_low_mma_ordinal=8,min_chain_peak=8)
OLD='''  auto load_a=[&](int slot,auto& al0,auto& al1,auto& ah0,auto& ah1) {
    auto dl0=lc.retile_D(al0),dl1=lc.retile_D(al1);
    auto dh0=hc.retile_D(ah0),dh1=hc.retile_D(ah1);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),dl0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),dl1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),dh0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),dh1);
  };'''
NEW='''  auto load_high=[&](int slot) {
    auto dh0=hc.retile_D(h00),dh1=hc.retile_D(h01);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),dh0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),dh1);
  };
  auto load_low=[&](int slot) {
    auto dl0=lc.retile_D(a00),dl1=lc.retile_D(a01);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),dl0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),dl1);
  };'''
MARKER='      // A G128 signed-high dot is at most8192 in magnitude.'
INSERT='''      // Same stage, after high/high and before reconstruction. Both N64
      // slices still reuse the same four A fragments; no additional LDSM.
      if constexpr(decltype(nb)::value==0) load_low(slot);
'''


def generated_header():
    text=generated_headers('o3')[0]
    call='    load_a(slot,a00,a01,h00,h01);'
    if any(text.count(x)!=1 for x in (OLD,call,MARKER)):
        raise ValueError('v89 load boundary drift')
    return text.replace(OLD,NEW).replace(call,'    load_high(slot);').replace(
        MARKER,INSERT+MARKER).replace('o3_grouped_cta_experiment','o3_late_low_experiment')


def operand_load_order(sass,symbol,live):
    """Map exact 4-GPR MMA A operands to LDSM writers, fail on unknown writers.

    Native IMMA.16864 A is four packed words. Tags are killed on register
    overwrite; MOV tags propagate. Not a timing/concurrency simulation.
    """
    loop=next(x for x in live['loops'] if x['kind']=='integer')
    ops=instructions(sass,symbol,loop)
    tags={};loads={};used={'S4':set(),'U4':set()};mma_count=0
    for pc,ins in ops:
        op=ins.split()[0]
        mma=re.match(r'IMMA\.16864\.(S4|U4)\.S4\s+R\d+, R(\d+)(?:\.reuse)?\.ROW,',ins)
        if mma:
            kind,start=mma[1],int(mma[2]);ids={tags.get(start+i) for i in range(4)}
            if None in ids or len(ids)!=1:
                raise ValueError('MMA A cannot be mapped to a unique LDSM at '+hex(pc))
            used[kind].update(ids);mma_count+=1
        dest=re.match(r'\S+\s+R(\d+)(?:\.\w+)?\s*,(.*)',ins)
        if not dest:continue
        first=int(dest[1]);rest=dest[2]
        width=4 if op.startswith(('LDSM.','IMMA.')) or (
            op.startswith(('LDG','LDS','LDL')) and '.128' in op) else 2 if (
            op.startswith(('LDG','LDS','LDL')) and '.64' in op) or op=='CS2R' or (
            op.startswith('IMAD.WIDE')) else 1
        value=None
        if op.startswith('LDSM.'):
            if not op.endswith('.4'):raise ValueError('expected x4 LDSM')
            value=pc;loads[pc]=dict(pc=hex(pc),mma_before=mma_count,instruction=ins)
        elif op=='MOV':
            src=re.match(r'\s*R(\d+)(?:\.reuse)?\s*$',rest)
            if src:value=tags.get(int(src[1]))
        for r in range(first,first+width):tags[r]=value
    if mma_count!=64 or any(len(s)!=4 for s in used.values()) or used['S4']&used['U4']:
        raise ValueError('expected distinct four high and four low A loads')
    result={kind:[loads[pc] for pc in sorted(ids)] for kind,ids in used.items()}
    result['first_low_mma_ordinal']=min(x['mma_before'] for x in result['U4'])
    return result


def cost_gate(old,new,order,chains):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer')
    counts=b['opcode_counts'];count=lambda p:sum(v for op,v in counts.items() if op.startswith(p))
    ratio=b['static_instructions']/a['static_instructions']
    checks=dict(work=ratio<=LIMITS['max_work_ratio'],
        allocation=new['allocated_gpr']<=LIMITS['max_allocated_gpr'],
        hot_local=count('LDL')+count('STL')<=LIMITS['max_hot_local'],
        native_math=count('IMMA.16864.S4.S4')==count('IMMA.16864.U4.S4')==32,
        same_supply=count('LDSM.')==16 and count('LDGSTS')==9,
        same_barrier=count('BAR')==1,
        low_loads_really_delayed=order['first_low_mma_ordinal']>=LIMITS['min_first_low_mma_ordinal'],
        chain_parallelism=chains['peak_started_not_finished_chains']>=LIMITS['min_chain_peak'])
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],work_ratio=ratio,
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
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_bytes(f.read_bytes())
    (out/(STEM+'_generated.cuh')).write_text(generated_header())
    sources=set(prior['sources'])|{'csrc/sm80/roof_o3_late_low_probe.cu',
        'scripts/probe_o3_late_low_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/inspect_o78_register_liveness.py','scripts/probe_roof_fullk_integer_codegen.py',
        'scripts/compare_a100_codegen.py'}
    r=dict(scope='O3_same_G128_low_load_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],limits=LIMITS,
        nvcc=version,cutlass_commit=commit,production_default_changed=False,changed_semantics=False,
        cta_tile=[64,128,128],threads=128,stages=3,shared_bytes=50688,partial_registers=32,
        conversion_changed=False,new_candidate_GPU_executed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd);(out/'progress.json').write_text(json.dumps(r,indent=2)+'\n')
        with (out/name).open('w') as f:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o3_late_low_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+['-O3','-lineinfo',src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-O3','-lineinfo',src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s in (CONTROL,SYMBOL):
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX native math/supply missing')
    control=compare((baseline/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    orders={s:operand_load_order(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    chains={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    gate=cost_gate(live[CONTROL],live[SYMBOL],orders[SYMBOL],chains[SYMBOL])
    gate['control_encoding_unchanged']=control['passed'];gate['passed'] &= control['passed']
    r.update(entries=es,liveness=live,load_order=orders,chains=chains,cost_gate=gate,
        control_comparison=control,cubin_sha256=sha(cubin),artifact_sha256={f.name:sha(f)
            for f in out.iterdir() if f.is_file() and f.name not in ('codegen.json','progress.json')})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native INT4/control audit failed')
    print(json.dumps(dict(cost_gate=gate,liveness=live,load_order=orders),indent=2),flush=True)
    print('PASSED: actual resources, safety, numeric and full24 pair next' if gate['passed'] else
        'FAILED: stop without GPU timing or neighboring schedule sweep',flush=True)


if __name__=='__main__':main()

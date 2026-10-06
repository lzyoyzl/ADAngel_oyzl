#!/usr/bin/env python3
"""v97: exactly two next-N64 startup chains, eight extra short-lived GPRs.

Keep the v96 B0/B1 lifetime ordering, but the next two high0 results no longer
depend on reading four old partial values first. One fixed prologue, no chain
count/stage/tile sweep. Same old64 native MMAs and16 LDSM per G128.
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
from probe_interleaved_tail_codegen import ROOT, CONFIG as OLD, generated_header as tail_header
from probe_grouped_cta_codegen import checked as grouped_checked
from probe_roof_fullk_integer_codegen import static_entries

CONFIG={kind:dict(cfg,symbol=f'adangel_roof_{kind}_tail_lookahead_candidate',
    stem=f'{kind}_tail_lookahead') for kind,cfg in OLD.items()}


def generated_header(kind):
    source=tail_header(kind)
    high='h00' if kind=='o3' else 'h0'
    b='b00' if kind=='o3' else 'b0'
    boundary='      load_b_half0(cute::_1{});\n'
    startup=f'''      //Two additional startup chains; only8 extra logical partial slots.
      //They do not alias any of the32 unfinished old partial registers.
      auto next_start=cute::make_tensor<int>(cute::make_shape(cute::_4{{}},cute::_2{{}}));
      cute::clear(next_start);
      o1_static_for<0,2>([&](auto ni) {{
        auto early=next_start(cute::_,ni);
        cute::gemm(HA{{}},early,{high}(cute::_,cute::_0{{}},cute::_0{{}}),{b}(cute::_,ni,cute::_0{{}}),early);
      }});
'''
    old=f'''          //Same four register slots; do not wait for the other seven chains.
          cute::clear(p);
          cute::gemm(HA{{}},p,{high}(cute::_,mi,cute::_0{{}}),{b}(cute::_,ni,cute::_0{{}}),p);
'''
    new=f'''          //Transfer the two early results only after consuming all old values.
          if constexpr (decltype(mi)::value==0 && decltype(ni)::value<2) {{
            cute::copy(next_start(cute::_,ni),p);
          }} else {{
            cute::clear(p);
            cute::gemm(HA{{}},p,{high}(cute::_,mi,cute::_0{{}}),{b}(cute::_,ni,cute::_0{{}}),p);
          }}
'''
    if source.count(boundary)!=1 or source.count(old)!=1:
        raise ValueError('v96 startup/tail source boundary drift')
    source=source.replace(boundary,boundary+startup).replace(old,new)
    return source.replace(kind+'_interleaved_tail_experiment',kind+'_tail_lookahead_experiment')


def schedule_summary(schedule):
    #Chain IDs are assigned by first SASS appearance, NOT logical N positions.
    starts=[s for s in schedule['schedule'] if s['stage']==1]
    if len(starts)!=16:raise ValueError('all16 output chains required')
    return dict(ninth_started_chain_first_mma_ordinal=starts[8]['ordinal'],
        peak_started_not_finished_chains=schedule['peak_started_not_finished_chains'],
        interpretation='PC-ordered chain IDs, not logical N coordinates, dynamic inflight count or speedup')


def worth_runtime(live,schedule):
    counts=next(x for x in live['loops'] if x['kind']=='integer')['opcode_counts']
    local=sum(v for op,v in counts.items() if op.split('.')[0] in ('LDL','STL'))
    return (live['allocated_gpr']<=168 and local<=2
        and counts.get('IMMA.16864.S4.S4')==counts.get('IMMA.16864.U4.S4')==32
        and counts.get('LDSM.16.M88.4')==16
        and 9<=schedule['peak_started_not_finished_chains']<=10)


def checked(directory,kind):
    cfg=CONFIG[kind];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('lookahead source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('lookahead artifact drift: '+name)
    if (directory/(cfg['stem']+'_generated.cuh')).read_text()!=generated_header(kind):
        raise ValueError('generated lookahead header drift')
    if not r['control_comparison']['passed'] or r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/cg copy failed')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();cfg=CONFIG[args.kind];out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    baseline=ROOT/cfg['baseline']
    if args.kind=='o3':grouped_checked(baseline,'o3')
    prior=json.loads((baseline/'codegen.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('best source drift: '+name)
    if sha(baseline/(cfg['old_stem']+'.cubin'))!=prior['cubin_sha256']:
        raise ValueError('best cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    (out/(cfg['stem']+'_generated.cuh')).write_text(generated_header(args.kind))
    sources=set(prior['sources'])|{'scripts/probe_tail_lookahead_codegen.py',
        'scripts/probe_interleaved_tail_codegen.py','scripts/inspect_eight_chain_schedule.py',
        f"csrc/sm80/roof_{args.kind}_tail_lookahead_probe.cu"}
    r=dict(scope='v97_fixed_two_chain_startup_compile_gate',kind=args.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=128,stages=cfg['stages'],shared_bytes=cfg['shared'],
        partial_registers=40,source_max_chains=10,extra_startup_chains=2,group_m=cfg['group_m'],
        changed_semantics=False,production_default_changed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/f"csrc/sm80/roof_{args.kind}_tail_lookahead_probe.cu")]
    cubin=out/(cfg['stem']+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(cfg['stem']+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],cfg['stem']+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(cfg['stem']+'.sass')).read_text();ptx=(out/(cfg['stem']+'.ptx')).read_text()
    symbols={cfg['control'],cfg['symbol']}
    entries=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    for symbol in symbols:
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native INT4/cg missing')
    lives={s:analyze((out/'liveness.txt').read_text(),s) for s in symbols}
    schedules={s:trace(sass,s,lives[s]) for s in symbols}
    r.update(entries=entries,liveness=lives,schedules=schedules,
        ordering={s:schedule_summary(schedules[s]) for s in symbols},cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['old_stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'))
    r['worth_runtime_validation']=worth_runtime(lives[cfg['symbol']],r['ordering'][cfg['symbol']])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out,args.kind)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],ordering=r['ordering'],liveness=lives),indent=2),flush=True)


if __name__=='__main__':main()

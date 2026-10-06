#!/usr/bin/env python3
"""v96: reuse each finished N64 chain for the following N64 chain immediately.

Unlike v92, retain32 partial registers/eight chains, the old tile/copies/stages
and all group arithmetic. B0 is overwritten only after every old low0 MMA;
B1 is overwritten only after every old low1 MMA. Audit the compiled dataflow:
the ninth chain must start before the eighth old chain finishes. No sweep.
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
from probe_grouped_cta_codegen import ROOT, generated_headers, checked as grouped_checked
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

CONFIG = {
    'o3': dict(baseline='reports/o378_roof_v89_o3_codegen',
        control='adangel_roof_o3_grouped_cta_candidate', old_stem='o3_grouped_cta',
        symbol='adangel_roof_o3_interleaved_tail_candidate', stem='o3_interleaved_tail',
        shared=50688, stages=3, group_m=8),
    'o78': dict(baseline='reports/o378_roof_v78_codegen',
        control='adangel_roof_o78_eight_chain_candidate', old_stem='o78_eight_chain',
        symbol='adangel_roof_o78_interleaved_tail_candidate', stem='o78_interleaved_tail',
        shared=34304, stages=2, group_m=1),
}


def phase(atom, a, b):
    return f'''      o1_static_for<0,2>([&](auto mi) {{
        o1_static_for<0,4>([&](auto ni) {{
          auto p=partial(cute::_,mi,ni);
          cute::gemm({atom}{{}},p,{a}(cute::_,mi,cute::_0{{}}),{b}(cute::_,ni,cute::_0{{}}),p);
        }});
      }});
'''


def generated_header(kind):
    if kind not in CONFIG:
        raise ValueError('O3 or O7/O8 required')
    source = generated_headers('o3')[0] if kind == 'o3' else eight_header(
        (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    names = dict(a0='a00',a1='a01',h0='h00',h1='h01',b0='b00',b1='b01') if kind=='o3' else {
        name:name for name in ('a0','a1','h0','h1','b0','b1')}
    b0,b1 = names['b0'],names['b1']
    coefficient = ('s.factor[slot][cute::get<1>(coord)]' if kind=='o3' else
        's.activation_factors[slot][cute::get<0>(coord)]*\n'
        '                                  s.weight_factors[slot][cute::get<1>(coord)]')
    code = f'''    {{
      //32 partial registers, eight chains. Recycle each chain only AFTER
      //all four old output contributions have been consumed.
      auto partial=cute::make_tensor<int>(cute::make_shape(cute::_4{{}},cute::_2{{}},cute::_4{{}}));
      cute::clear(partial);
      auto load_b_half0=[&](auto nb) {{
        auto destination=bc.retile_D({b0});
        cute::copy(SCopy{{}},bc.partition_S(tile_b(slot,nb,cute::_0{{}})),destination);
      }};
      auto load_b_half1=[&](auto nb) {{
        auto destination=bc.retile_D({b1});
        cute::copy(SCopy{{}},bc.partition_S(tile_b(slot,nb,cute::_1{{}})),destination);
      }};
      load_b_half0(cute::_0{{}});load_b_half1(cute::_0{{}});
'''
    code += phase('HA',names['h0'],b0)+phase('HA',names['h1'],b1)
    code += '      o1_static_for<0,32>([&](auto i) {partial(i)*=16;});\n'
    code += phase('LA',names['a0'],b0)
    code += f'''      //Old B0 is now dead; old B1 remains live until each old low1.
      //No extra load, new buffer, K stage or global payload is introduced.
      load_b_half0(cute::_1{{}});
      o1_static_for<0,2>([&](auto mi) {{
        o1_static_for<0,4>([&](auto ni) {{
          auto p=partial(cute::_,mi,ni);
          cute::gemm(LA{{}},p,{names['a1']}(cute::_,mi,cute::_0{{}}),{b1}(cute::_,ni,cute::_0{{}}),p);
          o1_static_for<0,4>([&](auto vi) {{
            const auto coord=coords(vi,mi,ni);
            const int coefficient={coefficient};
            acc(vi,mi,ni)+=partial(vi,mi,ni)*coefficient;
          }});
          //Same four register slots; do not wait for the other seven chains.
          cute::clear(p);
          cute::gemm(HA{{}},p,{names['h0']}(cute::_,mi,cute::_0{{}}),{b0}(cute::_,ni,cute::_0{{}}),p);
        }});
      }});
      //All old B1 reads have finished; replace it by the second N64 panel.
      load_b_half1(cute::_1{{}});
'''
    code += phase('HA',names['h1'],b1)
    code += '      o1_static_for<0,32>([&](auto i) {partial(i)*=16;});\n'
    code += phase('LA',names['a0'],b0)+phase('LA',names['a1'],b1)
    code += f'''      o1_static_for<0,2>([&](auto mi) {{
        o1_static_for<0,4>([&](auto ni) {{
          auto full_ni=cute::_4{{}}+ni;
          o1_static_for<0,4>([&](auto vi) {{
            const auto coord=coords(vi,mi,full_ni);
            const int coefficient={coefficient};
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
          }});
        }});
      }});
    }}'''
    begin_marker='    o1_static_for<0,2>([&](auto nb) {\n'
    end_marker='    });\n  }\n'+('  auto final_value=' if kind=='o3' else '  // Reuse the 64 INT32')
    if source.count(begin_marker)!=1 or source.count(end_marker)!=1:
        raise ValueError('best eight-chain body boundary drift')
    begin,end=source.index(begin_marker),source.index(end_marker)+len('    });')
    source=source[:begin]+code+source[end:]
    old_namespace='o3_grouped_cta_experiment' if kind=='o3' else 'o78_eight_chain_experiment'
    return source.replace(old_namespace,kind+'_interleaved_tail_experiment')


def ordering_summary(schedule):
    """Static order only, not a claim about simultaneous hardware execution."""
    chains=schedule['chains']
    if len(chains)!=16:
        raise ValueError('all16 output chains required')
    old_last=max(int(c['end_pc'],16) for c in chains[:8])
    next_first=min(int(c['start_pc'],16) for c in chains[8:])
    first_ordinal=next(s['ordinal'] for s in schedule['schedule'] if s['chain']==8 and s['stage']==1)
    return dict(first_next_slice_mma_ordinal=first_ordinal,
        next_slice_starts_before_all_old_mma_finish=next_first<old_last,
        peak_started_not_finished_chains=schedule['peak_started_not_finished_chains'],
        interpretation='static dataflow order, not hardware inflight count or predicted speedup')


def worth_runtime(live,ordering):
    counts=next(x for x in live['loops'] if x['kind']=='integer')['opcode_counts']
    hot_local=sum(v for op,v in counts.items() if op.split('.')[0] in ('LDL','STL'))
    return (live['allocated_gpr']<=168 and hot_local<=2
        and counts.get('IMMA.16864.S4.S4')==counts.get('IMMA.16864.U4.S4')==32
        and counts.get('LDSM.16.M88.4')==16
        and ordering['next_slice_starts_before_all_old_mma_finish']
        and ordering['peak_started_not_finished_chains']<=8)


def checked(directory,kind):
    cfg=CONFIG[kind];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('tail source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('tail artifact drift: '+name)
    if (directory/(cfg['stem']+'_generated.cuh')).read_text()!=generated_header(kind):
        raise ValueError('generated tail body drift')
    if not r['control_comparison']['passed'] or r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
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
    sources=set(prior['sources'])|{'scripts/probe_interleaved_tail_codegen.py',
        'scripts/inspect_eight_chain_schedule.py',f"csrc/sm80/roof_{args.kind}_interleaved_tail_probe.cu"}
    r=dict(scope='v96_eight_slot_N64_tail_pipeline_compile_gate',kind=args.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=128,stages=cfg['stages'],shared_bytes=cfg['shared'],
        partial_registers=32,independent_chains=8,group_m=cfg['group_m'],
        changed_semantics=False,production_default_changed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/f"csrc/sm80/roof_{args.kind}_interleaved_tail_probe.cu")]
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
            raise ValueError('same-entry native INT4/copy missing')
    live_text=(out/'liveness.txt').read_text()
    lives={s:analyze(live_text,s) for s in symbols}
    schedules={s:trace(sass,s,lives[s]) for s in symbols}
    r.update(entries=entries,liveness=lives,schedules=schedules,
        ordering={s:ordering_summary(schedules[s]) for s in symbols},cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['old_stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'))
    r['worth_runtime_validation']=worth_runtime(lives[cfg['symbol']],r['ordering'][cfg['symbol']])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out,args.kind)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],ordering=r['ordering'],liveness=lives),indent=2),flush=True)


if __name__=='__main__':main()

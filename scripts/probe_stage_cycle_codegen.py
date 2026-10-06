#!/usr/bin/env python3
"""v93: fixed cyclic SMEM slots, unchanged G128 math/MMA/guard.

Unroll exactly one2/3-stage ring, not all32 groups. A compile gate rejects
small/no work reduction, extra hot-loop local memory or greater register use.
No runtime performance claim is inferred from the static gate.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import ROOT, generated_headers, checked as grouped_checked
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

CONFIG = {
    'o3': dict(baseline='reports/o378_roof_v89_o3_codegen',
        control='adangel_roof_o3_grouped_cta_candidate', old_stem='o3_grouped_cta',
        symbol='adangel_roof_o3_stage_cycle_candidate', stem='o3_stage_cycle',
        shared=50688, stages=3, group_m=8),
    'o78': dict(baseline='reports/o378_roof_v78_codegen',
        control='adangel_roof_o78_eight_chain_candidate', old_stem='o78_eight_chain',
        symbol='adangel_roof_o78_stage_cycle_candidate', stem='o78_stage_cycle',
        shared=34304, stages=2, group_m=1),
}


def cycle_groups(stages):
    """Device schedule oracle: group, fixed slot, future group/slot or None."""
    if stages not in (2, 3):
        raise ValueError('only original2/3-stage rings')
    return [(g, g % stages, (g + stages - 1, (g + stages - 1) % stages)
             if g + stages - 1 < 32 else None) for g in range(32)]


def generated_header(kind):
    if kind not in CONFIG:
        raise ValueError('O3 or O7/O8 required')
    stages = CONFIG[kind]['stages']
    source = generated_headers('o3')[0] if kind == 'o3' else eight_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    start = '  for(int group=0;group<'
    stop = '\n  }\n  auto final_value=' if kind == 'o3' else '\n  }\n  // Reuse the 64 INT32'
    consumer = '    load_a(slot,' if kind == 'o3' else '    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),'
    assert source.count(start) == source.count(stop) == 1
    lo, hi = source.index(start), source.index(stop) + len('\n  }')
    old_loop = source[lo:hi]
    pos = old_loop.index(consumer)
    math = old_loop[pos:-len('\n  }')]
    wait = ('    if constexpr(decltype(future)::value)\n'
            '      asm volatile("cp.async.wait_group 1;" ::: "memory");\n'
            '    else asm volatile("cp.async.wait_group 0;" ::: "memory");\n'
            if kind == 'o3' else
            '    asm volatile("cp.async.wait_group 0;" ::: "memory");\n')
    args = 'a,w,ws,m,n,k' if kind == 'o3' else 'a,w,af,wf,m,n,k'
    # O3 retains the uint8 pointer ABI; its factor prefetch bitcasts internally.
    offset = stages - 1
    core = ('  auto process_group=[&](auto slot,int group,auto future) {\n' + wait +
        '    __syncthreads(); // Same all-reader stage release as the control.\n'
        '    if constexpr(decltype(future)::value)\n'
        f'      prefetch(s,(decltype(slot)::value+{offset})%{stages},group+{offset},{args});\n' +
        math + '\n  };\n')
    # Leave32%3=2 groups as drain, and the final2-stage pair as explicit drain.
    cycles = 10 if stages == 3 else 15
    core += ('  #pragma unroll 1\n'
        f'  for(int cycle=0;cycle<{cycles};++cycle) {{\n'
        f'    o1_static_for<0,{stages}>([&](auto slot) {{\n'
        f'      process_group(slot,cycle*{stages}+decltype(slot)::value,cute::Int<1>{{}});\n'
        '    });\n  }\n')
    core += ('  process_group(cute::Int<0>{},30,cute::Int<' + ('0' if stages == 3 else '1') + '>{});\n'
             '  process_group(cute::Int<1>{},31,cute::Int<0>{});')
    old_namespace = 'o3_grouped_cta_experiment' if kind == 'o3' else 'o78_eight_chain_experiment'
    return (source[:lo] + core + source[hi:]).replace(old_namespace, kind + '_stage_cycle_experiment')


def cycle_liveness(text, symbol, stages):
    """Exact backward-branch cycle, normalized by the2/3 original G128s.

    Do not feed an unrolled loop into the old64-MMA-per-group parser.
    Prologue/drain/fallback remain separate and are included in whole entry ISA.
    """
    blocks = re.split(r'(?=^//-+ \.text\.)', text, flags=re.M)
    block = next((b for b in blocks if re.match(r'//-+ \.text\.' + re.escape(symbol) + r'\s', b)), None)
    if block is None:
        raise ValueError('exact kernel missing')
    allocated = re.search(r'SHI_REGISTERS=(\d+)', block)
    if allocated is None:
        raise ValueError('allocated register metadata missing')
    ins, labels, pending = [], {}, []
    for line in block.splitlines():
        label = re.match(r'\s*(\.L_[A-Za-z0-9_]+):', line)
        if label:
            pending.append(label[1])
        item = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|', line)
        if item:
            pc = int(item[1], 16)
            for name in pending:
                labels[name] = pc
            pending.clear()
            ins.append(dict(pc=pc, instruction=item[2].strip(), live_gpr=int(item[3])))
    loops = []
    for item in ins:
        target = re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)', item['instruction'])
        if not target or target[1] not in labels or labels[target[1]] >= item['pc']:
            continue
        region = [i for i in ins if labels[target[1]] <= i['pc'] <= item['pc']]
        counts = Counter(re.sub(r'^@!?[A-Z0-9]+\s+', '', i['instruction']).split()[0] for i in region)
        if (sum(v for op,v in counts.items() if op.startswith('IMMA.')) != 64 * stages
                or counts.get('I2F', 0)):
            continue
        loops.append(dict(kind='integer', groups_per_iteration=stages,
            begin_pc=hex(region[0]['pc']), end_pc=hex(region[-1]['pc']),
            static_instructions=len(region), normalized_instructions_per_group=len(region)/stages,
            max_live_gpr=max(i['live_gpr'] for i in region), opcode_counts=dict(sorted(counts.items()))))
    if len(loops) != 1:
        raise ValueError('expected exactly one rolled2/3-group integer cycle')
    return dict(symbol=symbol, allocated_gpr=int(allocated[1]), loops=loops,
                interpretation='static_normalized_mainloop_not_dynamic_work_or_performance')


def worth_runtime(control, candidate, stages):
    old = next(x for x in control['loops'] if x['kind'] == 'integer')
    new = candidate['loops'][0]
    counts = new['opcode_counts']
    no_hot_local = not any(v for op,v in counts.items() if op.split('.')[0] in ('LDL','STL'))
    math_preserved = (counts.get('IMMA.16864.S4.S4') == counts.get('IMMA.16864.U4.S4') == 32*stages
        and counts.get('LDSM.16.M88.4') == 16*stages)
    meaningful = (new['normalized_instructions_per_group'] <= old['static_instructions']*.95
                  or candidate['allocated_gpr'] <= control['allocated_gpr']-16)
    return (no_hot_local and math_preserved and meaningful
            and candidate['allocated_gpr'] <= control['allocated_gpr'])


def checked(directory, kind):
    cfg = CONFIG[kind]; sha = lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    r = json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('stage-cycle source drift: ' + name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError('stage-cycle artifact drift: ' + name)
    if (directory/(cfg['stem']+'_generated.cuh')).read_text() != generated_header(kind):
        raise ValueError('stage-cycle generated body drift')
    if not r['control_comparison']['passed'] or r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1']
               for e in r['entries'].values()):
        raise ValueError('same-entry native math/copy failed')
    if r['worth_runtime_validation'] != worth_runtime(
            r['liveness'][cfg['control']],r['liveness'][cfg['symbol']],cfg['stages']):
        raise ValueError('compile-gate verdict drift')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=CONFIG, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); cfg = CONFIG[args.kind]; out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    baseline = ROOT/cfg['baseline']
    if args.kind == 'o3':
        grouped_checked(baseline,'o3')
    prior = json.loads((baseline/'codegen.json').read_text())
    sha = lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    for name,digest in prior['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('best source drift: ' + name)
    if sha(baseline/(cfg['old_stem']+'.cubin')) != prior['cubin_sha256']:
        raise ValueError('best cubin drift')
    cuda = Path('/usr/local/cuda-12.8/bin'); cutlass = ROOT/'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):
            (out/f.name).write_text(f.read_text())
    (out/(cfg['stem']+'_generated.cuh')).write_text(generated_header(args.kind))
    sources = set(prior['sources']) | {'scripts/probe_stage_cycle_codegen.py',
        f"csrc/sm80/roof_{args.kind}_stage_cycle_probe.cu"}
    r = dict(scope='v93_fixed_stage_cycle_compile_gate', kind=args.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)}, commands=[], nvcc=version, cutlass_commit=commit,
        cta_tile=[64,128,128], threads=128, stages=cfg['stages'], shared_bytes=cfg['shared'],
        partial_registers=32, independent_chains=8, group_m=cfg['group_m'],
        changed_semantics=False, production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(command, name):
        r['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags = [str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo',
        '-arch=sm_80','-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),
        '-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/f"csrc/sm80/roof_{args.kind}_stage_cycle_probe.cu")]
    cubin = out/(cfg['stem']+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(cfg['stem']+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],cfg['stem']+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass = (out/(cfg['stem']+'.sass')).read_text(); ptx = (out/(cfg['stem']+'.ptx')).read_text()
    symbols = {cfg['control'],cfg['symbol']}
    entries = static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    for symbol in symbols:
        body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                    if b.startswith('.visible .entry '+symbol+'('))
        if not all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native PTX math/copy missing')
    live = (out/'liveness.txt').read_text()
    control = analyze(live,cfg['control'])
    candidate = cycle_liveness(live,cfg['symbol'],cfg['stages'])
    r.update(entries=entries, liveness={cfg['control']:control,cfg['symbol']:candidate},
        cubin_sha256=sha(cubin), control_comparison=compare(
            (baseline/(cfg['old_stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'))
    r['worth_runtime_validation'] = worth_runtime(control,candidate,cfg['stages'])
    r['artifact_sha256'] = {f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name != 'codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n'); checked(out,args.kind)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],liveness=r['liveness']),indent=2),flush=True)


if __name__ == '__main__':
    main()

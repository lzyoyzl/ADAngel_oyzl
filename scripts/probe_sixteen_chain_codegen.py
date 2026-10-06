#!/usr/bin/env python3
"""v92: sixteen independent MMA chains in the same64x128 CTA.

One fixed scheduling experiment, not a tile/stage/register-cap sweep. Both
N64 slices are now live together. A/B work and group arithmetic are unchanged;
the explicit tradeoff is64 partial registers and at least2 resident CTAs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import ROOT, generated_headers, checked as grouped_checked
from probe_o78_eight_chain_codegen import MERGED, generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

CONFIG = {
    'o3': dict(baseline='reports/o378_roof_v89_o3_codegen',
        control='adangel_roof_o3_grouped_cta_candidate', old_stem='o3_grouped_cta',
        symbol='adangel_roof_o3_sixteen_chain_candidate', stem='o3_sixteen_chain',
        shared=50688, stages=3, group_m=8),
    'o78': dict(baseline='reports/o378_roof_v78_codegen',
        control='adangel_roof_o78_eight_chain_candidate', old_stem='o78_eight_chain',
        symbol='adangel_roof_o78_sixteen_chain_candidate', stem='o78_sixteen_chain',
        shared=34304, stages=2, group_m=1),
}


def generated_header(kind):
    if kind not in CONFIG:
        raise ValueError('O3 or O7/O8 required')
    source = generated_headers('o3')[0] if kind == 'o3' else eight_header(
        (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    owner = 'O3' if kind == 'o3' else 'O78'
    old = f'using SliceMma={owner}::O3AmpereConfig<64,64,128,false,2>::Mma;'
    assert source.count(old) == 1
    source = source.replace(old, 'using SliceMma=C::Mma;')
    # Only B is widened; A still contains both original K64 halves.
    old = 'cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(nb,half)'
    assert source.count(old) == 1
    source = source.replace(old,
        'cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(cute::_0{},half)')
    bname = 'b00' if kind == 'o3' else 'b0'
    old = f'static_assert(decltype(cute::size<1>({bname}))::value==4);'
    assert source.count(old) == 1
    source = source.replace(old, old.replace('==4', '==8'))
    merged = (MERGED.replace('Eight independent chains: two M atoms x four N atoms.',
        'Sixteen independent chains: two M atoms x eight N atoms.')
        .replace('Same32 logical partial registers; retain all original A/B reuse.',
            '64 logical partial registers; retain all original A/B reuse.')
        .replace('cute::_4{},cute::_2{},cute::_4{}', 'cute::_4{},cute::_2{},cute::_8{}')
        .replace('o1_static_for<0,4>([&](auto ni)', 'o1_static_for<0,8>([&](auto ni)')
        .replace('o1_static_for<0,32>', 'o1_static_for<0,64>')
        .replace('auto full_ni=nb*cute::_4{}+ni;', 'auto full_ni=ni;'))
    if kind == 'o3':
        merged = re.sub(r'\b(h0|h1|a0|a1|b0|b1)\b',
            lambda m: dict(h0='h00',h1='h01',a0='a00',a1='a01',b0='b00',b1='b01')[m[0]], merged)
        old = ('const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*\n'
               '                                  s.weight_factors[slot][cute::get<1>(coord)];')
        assert merged.count(old) == 1
        merged = merged.replace(old, 'const int coefficient=s.factor[slot][cute::get<1>(coord)];')
        loads = '      load_b(slot,cute::_0{},b00,b01);\n'
        end_marker = '    });\n  }\n  auto final_value='
        namespace = 'o3_grouped_cta_experiment'
    else:
        loads = ('      cute::copy(SCopy{},bc.partition_S(tile_b(slot,cute::_0{},cute::_0{})),bd0);\n'
                 '      cute::copy(SCopy{},bc.partition_S(tile_b(slot,cute::_0{},cute::_1{})),bd1);\n')
        end_marker = '    });\n  }\n  // Reuse the 64 INT32'
        namespace = 'o78_eight_chain_experiment'
    begin_marker = '    o1_static_for<0,2>([&](auto nb) {\n'
    assert source.count(begin_marker) == source.count(end_marker) == 1
    begin, end = source.index(begin_marker), source.index(end_marker) + len('    });')
    source = source[:begin] + '    {\n' + loads + merged + '    }' + source[end:]
    return source.replace(namespace, kind + '_sixteen_chain_experiment')


def worth_runtime(live):
    loop = next(x for x in live['loops'] if x['kind'] == 'integer')
    # Unlike address probes, allow the intended occupancy tradeoff. Large hot
    # spills would defeat the64-register dependency-hiding experiment.
    counts = loop['opcode_counts']
    return (live['allocated_gpr'] <= 248
        and sum(v for op,v in counts.items() if op.split('.')[0] in ('LDL','STL')) == 0
        and counts.get('IMMA.16864.S4.S4') == counts.get('IMMA.16864.U4.S4') == 32
        and counts.get('LDSM.16.M88.4') == 16)


def checked(directory, kind):
    cfg = CONFIG[kind]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory / 'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('sixteen-chain source drift: ' + name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory / name) != digest: raise ValueError('sixteen-chain artifact drift: ' + name)
    if (directory / (cfg['stem'] + '_generated.cuh')).read_text() != generated_header(kind):
        raise ValueError('generated sixteen-chain body drift')
    if not r['control_comparison']['passed'] or r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('encoded control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native math/copy failed')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); cfg = CONFIG[args.kind]; out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT): p.error('fresh repository output required')
    baseline = ROOT / cfg['baseline']
    if args.kind == 'o3': grouped_checked(baseline,'o3')
    prior = json.loads((baseline / 'codegen.json').read_text())
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    for name,digest in prior['sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('best source drift: ' + name)
    if sha(baseline / (cfg['old_stem'] + '.cubin')) != prior['cubin_sha256']:
        raise ValueError('best cubin drift')
    cuda = Path('/usr/local/cuda-12.8/bin'); cutlass = ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'): (out / f.name).write_text(f.read_text())
    (out / (cfg['stem'] + '_generated.cuh')).write_text(generated_header(args.kind))
    sources = set(prior['sources']) | {'scripts/probe_sixteen_chain_codegen.py',
        f"csrc/sm80/roof_{args.kind}_sixteen_chain_probe.cu"}
    r = dict(scope='v92_sixteen_chains_same_CTA_compile_gate',kind=args.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=128,stages=cfg['stages'],shared_bytes=cfg['shared'],
        partial_registers=64,independent_chains=16,group_m=cfg['group_m'],
        changed_semantics=False,production_default_changed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log: subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags = [str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo',
        '-arch=sm_80','-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),
        '-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/f"csrc/sm80/roof_{args.kind}_sixteen_chain_probe.cu")]
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
        body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native INT4/copy missing')
    live = (out/'liveness.txt').read_text()
    r.update(entries=entries,liveness={s:analyze(live,s) for s in sorted(symbols)},cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['old_stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'))
    r['worth_runtime_validation'] = worth_runtime(r['liveness'][cfg['symbol']])
    r['artifact_sha256'] = {f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n'); checked(out,args.kind)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],liveness=r['liveness']),indent=2),flush=True)


if __name__ == '__main__': main()

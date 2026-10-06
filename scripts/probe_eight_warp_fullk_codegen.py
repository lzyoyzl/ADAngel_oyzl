#!/usr/bin/env python3
"""v98: one fixed eight-warp full-K integer geometry, not a geometry sweep.

Preserve best O3 grouped traversal / O7-O8 natural traversal, all G128 math,
conversion, guards and stages. Halve final registers/thread, but account for
duplicated A fragment reads. Compilation gate precedes any candidate launch.
The v40 eight-warp experiment used a different, group-FP32 implementation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze as old_analyze
from probe_grouped_cta_codegen import ROOT, generated_headers, remap, checked as grouped_checked
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

CONFIG = {
    'o3': dict(baseline='reports/o378_roof_v89_o3_codegen',
        control='adangel_roof_o3_grouped_cta_candidate', old_stem='o3_grouped_cta',
        symbol='adangel_roof_o3_eight_warp_fullk_candidate', stem='o3_eight_warp_fullk',
        shared=50688, stages=3, group_m=8),
    'o78': dict(baseline='reports/o378_roof_v78_codegen',
        control='adangel_roof_o78_eight_chain_candidate', old_stem='o78_eight_chain',
        symbol='adangel_roof_o78_eight_warp_fullk_candidate', stem='o78_eight_warp_fullk',
        shared=34304, stages=2, group_m=1),
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('eight-warp source boundary drift: ' + old)
    return text.replace(old, new)


def generated_header(kind):
    if kind not in CONFIG:
        raise ValueError('O3 or O7/O8 required')
    source = generated_headers('o3')[0] if kind == 'o3' else eight_header(
        (ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    old_namespace = 'o3_grouped_cta_experiment' if kind == 'o3' else 'o78_eight_chain_experiment'
    source = source.replace(old_namespace, kind+'_eight_warp_fullk_experiment')
    alias = 'O3' if kind == 'o3' else 'O78'
    dual, stages = ('false',3) if kind == 'o3' else ('true',2)
    source = replace_once(source, f'using C={alias}::O3AmpereConfig<64,128,128,false,2,{dual},{stages}>;',
        f'using C={kind}_eight_warp_fallback::O3AmpereConfig<64,128,128,false,4,{dual},{stages}>;')
    #256 threads each copy16B. Keep every payload byte and factor panel once.
    source = replace_once(source, 'o1_static_for<0,2>([&](auto chunk)',
        'o1_static_for<0,1>([&](auto chunk)')
    source = replace_once(source, 'o1_static_for<0,4>([&](auto chunk)',
        'o1_static_for<0,2>([&](auto chunk)')
    if source.count('chunk*2048') != 2:
        raise ValueError('two copy strides required')
    source = source.replace('chunk*2048', 'chunk*4096')
    source = replace_once(source, f'using SliceMma={alias}::O3AmpereConfig<64,64,128,false,2>::Mma;',
        'using SliceMma=C::Mma; //Each warp owns its whole N32, no second N64 batch.')
    #Only B tile changes; A keeps M64/K64, and WM remains two.
    source = replace_once(source,
        'cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(nb,half)',
        'cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(nb,half)')
    source = replace_once(source, '    o1_static_for<0,2>([&](auto nb) {',
        '    { constexpr auto nb=cute::_0{}; //All N128 belongs to this warp geometry.')
    boundary = '    });\n  }\n  '+('auto final_value=' if kind=='o3' else '// Reuse the 64 INT32')
    source = replace_once(source, boundary,
        '    }\n  }\n  '+('auto final_value=' if kind=='o3' else '// Reuse the 32 INT32'))
    source = replace_once(source, 'auto full_ni=nb*cute::_4{}+ni;', 'auto full_ni=ni;')
    if kind=='o78':
        source = replace_once(source, 'static_assert(decltype(cute::size(acc))::value==64);',
            'static_assert(decltype(cute::size(acc))::value==32);')
    else:
        source = replace_once(source, 'static_assert(decltype(cute::size<1>(acc))::value==2);',
            'static_assert(decltype(cute::size<1>(acc))::value==2);\n'
            '  static_assert(decltype(cute::size(acc))::value==32);')
    return '//v98 fixed8 warp, full-K integer; no quantization or default change.\n'+source


def generated_fallback(kind):
    if kind not in CONFIG:
        raise ValueError('O3 or O7/O8 required')
    old = 'o3_row_scale_epilogue_experiment_v40' if kind=='o3' else 'o78_unsigned_payload_experiment_v40'
    text = (ROOT/f'csrc/sm80/{kind}_warp_geometry_probe.cuh').read_text()
    return remap(text,old,kind+'_eight_warp_fallback') if kind=='o3' else text.replace(
        old,kind+'_eight_warp_fallback')


def analyze(text, symbol, expected_mma=32):
    """Exact entry and two G128 loops; account for256 threads explicitly.

Do not modify the old64-MMA liveness parser used by immutable receipts.
"""
    functions = re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
    block = next((b for b in functions if re.match(
        r'//-+ \.text\.'+re.escape(symbol)+r'\s',b)),None)
    if block is None: raise ValueError('exact eight-warp kernel section missing')
    allocated = re.search(r'SHI_REGISTERS=(\d+)',block)
    if allocated is None: raise ValueError('register allocation metadata missing')
    instructions,labels,pending=[],{},[]
    for line in block.splitlines():
        label = re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label: pending.append(label[1])
        ins = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if ins:
            pc,op,live=int(ins[1],16),ins[2].strip(),int(ins[3])
            for name in pending: labels[name]=pc
            pending.clear();instructions.append(dict(pc=pc,instruction=op,live_gpr=live))
    if not instructions: raise ValueError('instruction count-mode liveness missing')
    loops=[]
    for ins in instructions:
        branch=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',ins['instruction'])
        if branch and branch[1] not in labels: raise ValueError('unresolved backward target')
        if not branch or labels[branch[1]]>=ins['pc']: continue
        region=[i for i in instructions if labels[branch[1]]<=i['pc']<=ins['pc']]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',i['instruction']).split()[0] for i in region)
        if sum(v for op,v in counts.items() if op.startswith('IMMA.'))!=expected_mma: continue
        maximum=max(i['live_gpr'] for i in region)
        loops.append(dict(kind='fp32_fallback' if counts.get('I2F',0) else 'integer',
            begin_pc=hex(region[0]['pc']),end_pc=hex(region[-1]['pc']),
            static_instructions=len(region),max_live_gpr=maximum,
            opcode_counts=dict(sorted(counts.items())),
            peak_pcs=[hex(i['pc']) for i in region if i['live_gpr']==maximum]))
    if len(loops)!=2 or {x['kind'] for x in loops}!={'integer','fp32_fallback'}:
        raise ValueError('one complete32-MMA integer and FP32 fallback loop required')
    return dict(symbol=symbol,allocated_gpr=int(allocated[1]),loops=loops,
        function_max_live_gpr=max(i['live_gpr'] for i in instructions),cta_threads=256,
        interpretation='static_binary_loop_counts_not_dynamic_cost_or_speedup')


def gate(control,candidate):
    old = next(x for x in control['loops'] if x['kind']=='integer')
    new = next(x for x in candidate['loops'] if x['kind']=='integer')
    c = new['opcode_counts']
    local = sum(v for op,v in c.items() if op.split('.')[0] in ('LDL','STL'))
    old_warps,new_warps=control['cta_threads']//32,candidate['cta_threads']//32
    ratio=(new['static_instructions']*new_warps)/(old['static_instructions']*old_warps)
    result=dict(registers_at_most128=candidate['allocated_gpr']<=128,
        hot_local_at_most2=local<=2,
        same_weighted_mma=(c.get('IMMA.16864.S4.S4')==c.get('IMMA.16864.U4.S4')==16
            and 32*new_warps==64*old_warps),
        ldsm_at_most12=c.get('LDSM.16.M88.4',0)<=12 and c.get('LDSM.16.M88.4',0)>0,
        weighted_static_instruction_ratio=ratio,weighted_growth_at_most10pct=ratio<=1.10,
        hot_local_instructions=local)
    result['passed']=all(result[key] for key in (
        'registers_at_most128','hot_local_at_most2','same_weighted_mma',
        'ldsm_at_most12','weighted_growth_at_most10pct'))
    result['runtime_resource_gate']='Still require >=2 resident CTAs/SM and16 active warps/SM before timings'
    return result


def checked(directory,kind):
    cfg=CONFIG[kind];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('eight-warp source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('eight-warp artifact drift: '+name)
    for suffix,body in (('_generated.cuh',generated_header(kind)),
                        ('_fallback_generated.cuh',generated_fallback(kind))):
        if (directory/(cfg['stem']+suffix)).read_text()!=body:raise ValueError('generated header drift')
    if not r['control_comparison']['passed'] or r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/cg copy failed')
    if gate(r['liveness'][cfg['control']],r['liveness'][cfg['symbol']])!=r['compile_gate']:
        raise ValueError('compile gate receipt drift')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind];out=a.output.resolve();baseline=ROOT/cfg['baseline']
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    if a.kind=='o3':grouped_checked(baseline,'o3')
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
    (out/(cfg['stem']+'_generated.cuh')).write_text(generated_header(a.kind))
    (out/(cfg['stem']+'_fallback_generated.cuh')).write_text(generated_fallback(a.kind))
    sources=set(prior['sources'])|{'scripts/probe_eight_warp_fullk_codegen.py',
        'scripts/inspect_o78_register_liveness.py',f'csrc/sm80/roof_{a.kind}_eight_warp_fullk_probe.cu',
        f'csrc/sm80/{a.kind}_warp_geometry_probe.cuh',
        'tests/cuda/validate_eight_warp_fullk_coordinates.cu'}
    r=dict(scope='v98_fixed_eight_warp_fullK_compile_gate',kind=a.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],threads=256,stages=cfg['stages'],shared_bytes=cfg['shared'],
        final_accumulator_registers_per_thread=32,partial_registers=32,group_m=cfg['group_m'],
        changed_semantics=False,production_default_changed=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    common=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coordinates=out/'validate_coordinates'
    run(common+[str(ROOT/'tests/cuda/validate_eight_warp_fullk_coordinates.cu'),'-o',str(coordinates)],
        'coordinates_build.log')
    run([str(coordinates)],'coordinates.log')
    r['host_cute_coordinates_passed']=True
    flags=common+['-lineinfo','-DADANGEL_FULLK_INTEGER=1',
        str(ROOT/f'csrc/sm80/roof_{a.kind}_eight_warp_fullk_probe.cu')]
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
    text=(out/'liveness.txt').read_text()
    lives={cfg['control']:old_analyze(text,cfg['control']),cfg['symbol']:analyze(text,cfg['symbol'])}
    r.update(entries=entries,liveness=lives,cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['old_stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'))
    r['compile_gate']=gate(lives[cfg['control']],lives[cfg['symbol']])
    r['worth_runtime_validation']=r['compile_gate']['passed']
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out,a.kind)
    print(json.dumps(dict(runtime_justified=r['worth_runtime_validation'],
        compile_gate=r['compile_gate'],liveness=lives),indent=2),flush=True)


if __name__=='__main__':main()

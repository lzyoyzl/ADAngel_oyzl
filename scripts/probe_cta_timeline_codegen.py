#!/usr/bin/env python3
"""v101 profiler instrumentation gate, NOT a performance optimization candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONFIG={
    'o3':dict(baseline='reports/o378_roof_v89_o3_codegen',stem='o3_grouped_cta',
        control='adangel_roof_o3_grouped_cta_candidate',symbol='adangel_roof_o3_cta_timeline',
        shared=50688,stages=3),
    'o78':dict(baseline='reports/o378_roof_v78_codegen',stem='o78_eight_chain',
        control='adangel_roof_o78_eight_chain_candidate',symbol='adangel_roof_o78_cta_timeline',
        shared=34304,stages=2),
}


def instrumentation_gate(old,new):
    """Same capacity and per-group work, not a claim of zero perturbation."""
    counts=lambda r:next(x for x in r['loops'] if x['kind']=='integer')['opcode_counts']
    a,b=counts(old),counts(new)
    local=lambda c:sum(v for op,v in c.items() if op.split('.')[0] in ('LDL','STL'))
    return dict(passed=(new['allocated_gpr']<=old['allocated_gpr']<=168
        and all(b.get(op,0)==a.get(op,0) for op in
            ('IMMA.16864.S4.S4','IMMA.16864.U4.S4','LDSM.16.M88.4','BAR.SYNC'))
        and local(b)<=local(a)),
        old_registers=old['allocated_gpr'],new_registers=new['allocated_gpr'],
        old_integer_instructions=next(x for x in old['loops'] if x['kind']=='integer')['static_instructions'],
        new_integer_instructions=next(x for x in new['loops'] if x['kind']=='integer')['static_instructions'],
        scope='compile_gate_actual_occupancy_and_Event_perturbation_still_require_runtime')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind];out=a.output.resolve();baseline=ROOT/cfg['baseline']
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    old=json.loads((baseline/'codegen.json').read_text())
    for name,digest in old['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('baseline source drift: '+name)
    cubin=baseline/(cfg['stem']+'.cubin')
    if sha(cubin)!=old['cubin_sha256']:raise ValueError('baseline cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_bytes(f.read_bytes())
    sources=set(old['sources'])|{'csrc/sm80/roof_cta_timeline.cuh',
        f'csrc/sm80/roof_{a.kind}_cta_timeline_probe.cu',
        'csrc/sm80/roof_cta_timeline_driver.cpp','scripts/probe_cta_timeline_codegen.py'}
    receipt=dict(scope='v101_CTA_warp_timeline_instrumentation_not_optimization',kind=a.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},baseline_cubin_sha256=old['cubin_sha256'],
        production_default_changed=False,new_performance_result=False,commands=[],
        timer_reference='https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#special-registers-globaltimer',
        nvcc=version,cutlass_commit=commit,config=cfg)
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/f'csrc/sm80/roof_{a.kind}_cta_timeline_probe.cu')]
    stem=a.kind+'_cta_timeline';binary=out/(stem+'.cubin')
    run(flags+['-cubin','-o',str(binary),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(stem+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(binary)],stem+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(binary)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(binary)],'liveness.txt')
    run(['g++','-O3','-std=c++17','-shared','-fPIC','-I/usr/local/cuda-12.8/include',
        str(ROOT/'csrc/sm80/roof_cta_timeline_driver.cpp'),'-lcuda','-o',str(out/'libcta_timeline.so')],'driver_build.log')
    sass=(out/(stem+'.sass')).read_text();ptx=(out/(stem+'.ptx')).read_text()
    symbols={cfg['control'],cfg['symbol']}
    entries=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in symbols}
    for s in symbols:
        if not (entries[s]['native_u4_s4'] and entries[s]['native_s4_s4']
                and not entries[s]['int8_mma'] and entries[s]['all_copies_bypass_l1']):
            raise ValueError('same-entry native INT4/copy gate failed')
    entry=next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
               if b.startswith('.visible .entry '+cfg['symbol']+'('))
    if '%globaltimer' not in entry or '%smid' not in entry:raise ValueError('timeline registers missing')
    receipt.update(entries=entries,liveness=live,cubin_sha256=sha(binary),
        control_comparison=compare((baseline/(cfg['stem']+'.sass')).read_text(),sass,'^'+cfg['control']+'$'),
        gate=instrumentation_gate(live[cfg['control']],live[cfg['symbol']]))
    if not receipt['control_comparison']['passed']:raise ValueError('old encoded control changed')
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(kind=a.kind,gate=receipt['gate'],control_comparison=receipt['control_comparison']),indent=2),flush=True)


if __name__=='__main__':main()

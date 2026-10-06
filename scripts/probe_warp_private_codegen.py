#!/usr/bin/env python3
"""v109 fixed warp-private in-place ring, not stage/warp/geometry enumeration.

Same CTA outputs/64 MMA/16 LDSM/32 partial registers and source math. Different
thread ownership is host-verified. Global payload reads double intentionally;
same total34304B SMEM. Remove all integer-path CTA barriers, not correctness
barriers. Only compile if original v78 receipts match; no default/native rebuild.
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
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reports/o378_roof_v78_codegen'
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_warp_private_candidate'
STEM='o78_warp_private'
SHARED=34304


def gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer')
    b=next(x for x in new['loops'] if x['kind']=='integer');c=b['opcode_counts']
    locals=sum(v for k,v in c.items() if k.startswith(('LDL','STL')))
    result=dict(same_native_MMA=c.get('IMMA.16864.S4.S4')==c.get('IMMA.16864.U4.S4')==32,
        same_LDSM=c.get('LDSM.16.M88.4')==16,no_hot_local=locals==0,
        registers_at_most168=new['allocated_gpr']<=168,
        no_integer_CTA_barrier=c.get('BAR.SYNC',0)==0,
        warp_sync_present=c.get('WARPSYNC',0)>=2,
        static_work_growth_at_most15pct=b['static_instructions']<=1.15*a['static_instructions'],
        static_work_ratio=b['static_instructions']/a['static_instructions'])
    result['passed']=all(v for k,v in result.items() if k!='static_work_ratio')
    result['runtime_gate']='actual>=3 CTAs and12 warps/SM before full24 timings; validate race/sync'
    return result


def checked(directory):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for path,digest in r['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('warp-private source drift: '+path)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest:raise ValueError('warp-private artifact drift')
    if not r['control_comparison']['passed'] or r['production_default_changed']:
        raise ValueError('old control/default drift')
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:
        raise ValueError('mapping proof missing')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4 and cg copy audit failed')
    if r['compile_gate']!=gate(r['liveness'][CONTROL],r['liveness'][SYMBOL]):
        raise ValueError('predeclared gate drift')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((BASELINE/'codegen.json').read_text())
    for path,digest in prior['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('v78 source drift')
    if sha(BASELINE/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in BASELINE.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    sources=set(prior['sources'])|{'scripts/probe_warp_private_codegen.py',
        'csrc/sm80/o78_warp_private_pipeline.cuh','csrc/sm80/roof_o78_warp_private_probe.cu',
        'tests/cuda/validate_warp_private_coordinates.cu'}
    r=dict(scope='v109_one_fixed_warp_private_in_place_compile_gate',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],warp_tile=[32,64,128],threads=128,private_buffer_depth=1,
        shared_bytes=SHARED,partial_registers=32,independent_chains=8,
        payload_global_copy_bytes_ratio=2,CTA_output_ownership_changed=True,
        G128_order_unchanged=True,source_math_and_guard_unchanged=True,production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    common=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coord=out/'validate_coordinates'
    run(common+['-O2',str(ROOT/'tests/cuda/validate_warp_private_coordinates.cu'),'-o',str(coord)],'coordinates_build.log')
    run([str(coord)],'coordinates.log');r['coordinates']=json.loads((out/'coordinates.log').read_text())
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:raise ValueError('host coordinate gate failed')
    flags=common+['-O3','-lineinfo',str(ROOT/'csrc/sm80/roof_o78_warp_private_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text();symbols={CONTROL,SYMBOL}
    r['entries']=static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols)
    for symbol in symbols:
        body=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        if not all(s in body for s in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry native MMA/copy missing')
    live=(out/'liveness.txt').read_text()
    r.update(liveness={s:analyze(live,s) for s in symbols},cubin_sha256=sha(cubin),
        control_comparison=compare((BASELINE/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'))
    r['compile_gate']=gate(r['liveness'][CONTROL],r['liveness'][SYMBOL])
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n');checked(out)
    print(json.dumps(dict(coordinates=r['coordinates'],compile_gate=r['compile_gate'],liveness=r['liveness']),indent=2),flush=True)


if __name__=='__main__':main()

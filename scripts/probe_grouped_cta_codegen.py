#!/usr/bin/env python3
"""v89: one fixed GROUP_M=8 ordering, retaining v79/v78 kernel math.

This is a CTA-coordinate experiment, not SMEM swizzling, a persistent kernel,
new quantizer, or a tile/stage sweep. Validate guards/fallback at logical IDs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o3_eight_chain_codegen import generated_header as o3_header
from probe_o78_eight_chain_codegen import generated_header as o78_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
REFERENCE='https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html'
CONFIG={
    'o3':dict(control='adangel_roof_o3_eight_chain_candidate',
        symbol='adangel_roof_o3_grouped_cta_candidate',stem='o3_grouped_cta',
        baseline='reports/o378_roof_v79_codegen',cubin='o3_eight_chain.cubin',
        shared=50688,stages=3),
    'o78':dict(control='adangel_roof_o78_eight_chain_candidate',
        symbol='adangel_roof_o78_grouped_cta_candidate',stem='o78_grouped_cta',
        baseline='reports/o378_roof_v78_codegen',cubin='o78_eight_chain.cubin',
        shared=34304,stages=2),
}


def mapping(x,y,columns,rows):
    """Exact host oracle of the device mapping, including the final group."""
    if min(columns,rows)<=0 or not (0<=x<columns and 0<=y<rows):
        raise ValueError('valid physical CTA/grid required')
    if columns%8==0 and rows%8==0:
        return (y%8)*(columns//8)+x//8, (y//8)*8+x%8
    linear=y*columns+x;span=8*columns;first=(linear//span)*8
    height=min(rows-first,8);within=linear-(first//8)*span
    return within//height,first+within%height


def remap(text,old_namespace,new_namespace):
    if text.count('namespace '+old_namespace+' {')!=1 or 'blockIdx.x' not in text or 'blockIdx.y' not in text:
        raise ValueError('coordinate source boundary drift')
    return (text.replace(old_namespace,new_namespace)
        .replace('blockIdx.x','roof_grouped_cta::tile().x')
        .replace('blockIdx.y','roof_grouped_cta::tile().y'))


def generated_headers(kind):
    if kind=='o78':
        body=o78_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
        fallback=(ROOT/'csrc/sm80/o78_unsigned_payload_candidate.cuh').read_text()
        return remap(body,'o78_eight_chain_experiment','o78_grouped_cta_experiment'), remap(
            fallback,'o78_unsigned_payload_experiment','o78_grouped_fallback')
    if kind=='o3':
        body=o3_header((ROOT/'csrc/sm80/o3_fullk_integer_probe.cuh').read_text())
        fallback=(ROOT/'csrc/sm80/o3_row_scale_epilogue_candidate.cuh').read_text()
        return remap(body,'o3_eight_chain_experiment','o3_grouped_cta_experiment'), remap(
            fallback,'o3_row_scale_epilogue_experiment','o3_grouped_fallback')
    raise ValueError('O3 or O7/O8 required')


def checked(directory,kind):
    cfg=CONFIG[kind];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((directory/'codegen.json').read_text())
    for p,digest in r['sources'].items():
        if sha(ROOT/p)!=digest:raise ValueError('grouped source drift: '+p)
    for p,digest in r['artifact_sha256'].items():
        if sha(directory/p)!=digest:raise ValueError('grouped artifact drift: '+p)
    if sha(directory/(cfg['stem']+'.cubin'))!=r['cubin_sha256']:
        raise ValueError('grouped cubin drift')
    for name,body in zip((cfg['stem']+'_generated.cuh',kind+'_grouped_fallback_generated.cuh'),generated_headers(kind)):
        if (directory/name).read_text()!=body:raise ValueError('generated grouped body drift')
    if not r['control_comparison']['passed'] or r['production_default_changed'] or r['changed_semantics']:
        raise ValueError('control/default/semantics mismatch')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=CONFIG,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=CONFIG[a.kind];out=a.output.resolve();baseline=ROOT/cfg['baseline']
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((baseline/'codegen.json').read_text())
    for name,digest in prior['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('best source drift: '+name)
    if sha(baseline/cfg['cubin'])!=prior['cubin_sha256']:raise ValueError('best cubin drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    # Include the previous complete candidate source so the control can be
    # verified against its original encoded SASS, not a renamed approximation.
    names=['o78_eight_chain_generated.cuh'] if a.kind=='o78' else [
        'o3_eight_chain_generated.cuh','o3_device_control_generated.cu','o3_factor_async_generated.cuh']
    for name in names:(out/name).write_text((baseline/name).read_text())
    for name,body in zip((cfg['stem']+'_generated.cuh',a.kind+'_grouped_fallback_generated.cuh'),generated_headers(a.kind)):
        (out/name).write_text(body)
    sources=set(prior['sources'])|{
        'csrc/sm80/roof_grouped_cta.cuh',f'csrc/sm80/roof_{a.kind}_grouped_cta_probe.cu',
        'scripts/probe_grouped_cta_codegen.py'}
    receipt=dict(scope='grouped_CTA_compile_gate_not_performance_acceptance',kind=a.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},baseline_cubin_sha256=prior['cubin_sha256'],
        reference=REFERENCE,group_m=8,cta_tile=[64,128,128],threads=128,stages=cfg['stages'],
        shared_bytes=cfg['shared'],changed_semantics=False,production_default_changed=False,
        nvcc=version,cutlass_commit=commit,commands=[])
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/f'csrc/sm80/roof_{a.kind}_grouped_cta_probe.cu');stem=cfg['stem'];cubin=out/(stem+'.cubin')
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/(stem+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],stem+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(stem+'.sass')).read_text();ptx=(out/(stem+'.ptx')).read_text()
    symbols={cfg['control'],cfg['symbol']}
    entries=static_entries(sass,'^(?:'+'|'.join(symbols)+')$',symbols)
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in symbols}
    for s in symbols:
        block=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in block for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX native math/copy missing')
    receipt.update(entries=entries,liveness=live,cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['cubin'].replace('.cubin','.sass'))).read_text(),sass,'^'+cfg['control']+'$'),
        artifact_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checked(out,a.kind)
    for s in symbols:
        loop=next(x for x in live[s]['loops'] if x['kind']=='integer')
        print(json.dumps(dict(symbol=s,registers=live[s]['allocated_gpr'],loop=loop),indent=2))
    print('Same-entry native INT4 and unchanged-control audit passed; no runtime result yet.')


if __name__=='__main__':main()

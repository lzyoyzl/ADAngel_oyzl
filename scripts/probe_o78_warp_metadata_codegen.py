#!/usr/bin/env python3
"""One measured-hotspot candidate: full-warp A-factor copies, same v78 math.

v80 NCU concentrated shared excessive wavefronts at the half-warp A-factor
cp.async. Use32 lanes x8B for A and another32 lanes x16B for W. No padding,
duplicate data, different factor arithmetic, or new source quantization.
The required .ca8B also changes caching of this tiny metadata load; this is
not claimed to be a pure lane-participation microbenchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header, SYMBOL as CONTROL
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
SYMBOL='adangel_roof_o78_warp_metadata_candidate'
OLD='''  const unsigned first=threadIdx.x*4;
  if(threadIdx.x<16) copy16(s.activation_factors[slot]+first,
      af+group*m+blockIdx.y*64+first);
  if(threadIdx.x<32) copy16(s.weight_factors[slot]+first,
      wf+group*n+blockIdx.x*128+first);
'''
NEW='''  // Full participating warps, same256B A +512B W per CTA/G128.
  if(threadIdx.x<32) {
    const unsigned first=threadIdx.x*2;
    copy_factor8(s.activation_factors[slot]+first,af+group*m+blockIdx.y*64+first);
  }
  if(threadIdx.x>=32 && threadIdx.x<64) {
    const unsigned first=(threadIdx.x-32)*4;
    copy16(s.weight_factors[slot]+first,wf+group*n+blockIdx.x*128+first);
  }
'''


def generated_header(source):
    text=eight_header(source)
    if text.count(OLD)!=1: raise ValueError('v78 metadata supply source drift')
    return text.replace(OLD,NEW).replace('o78_eight_chain_experiment','o78_warp_metadata_experiment')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((a.baseline/'codegen.json').read_text())
    for path,sha in prior['sources'].items():
        if digest(ROOT/path)!=sha:raise ValueError('v78 source drift: '+path)
    if digest(a.baseline/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':p.error('pinned toolchain required')
    out.mkdir(parents=True);source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header=out/'o78_warp_metadata_generated.cuh';header.write_text(generated_header(source))
    sources=set(prior['sources'])|{'csrc/sm80/roof_o78_warp_metadata_probe.cu','scripts/probe_o78_warp_metadata_codegen.py'}
    r=dict(scope='full_warp_factor_copy_compile_not_performance',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:digest(ROOT/s) for s in sorted(sources)},generated_header_sha256=digest(header),
        baseline_cubin_sha256=prior['cubin_sha256'],production_default_changed=False,
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=34304,
        metadata_bytes_per_cta_group=768,activation_factor_copy='warp0_ca8B',weight_factor_copy='warp1_cg16B',
        nvcc=version,cutlass_commit=commit,commands=[])
    def run(command,name):
        r['commands'].append(command)
        with (out/name).open('w') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    args=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/'csrc/sm80/roof_o78_warp_metadata_probe.cu')]
    cubin=out/'o78_warp_metadata.cubin'
    run(args+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(args+['-ptx','-o',str(out/'o78_warp_metadata.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_warp_metadata.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o78_warp_metadata.sass').read_text()
    entries=static_entries(sass,'^adangel_roof_o78_(?:eight_chain_candidate|warp_metadata_candidate)$',{CONTROL,SYMBOL})
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] for e in entries.values()):raise ValueError('native INT4 missing')
    block=next(b for b in re.split(r'(?=\.visible \.entry )',(out/'o78_warp_metadata.ptx').read_text()) if b.startswith('.visible .entry '+SYMBOL+'('))
    if not all(s in block for s in ('cp.async.ca.shared.global','cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):raise ValueError('PTX contract failed')
    r.update(cubin_sha256=digest(cubin),entries=entries,
        control_comparison=compare((a.baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'),
        liveness={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)})
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    if not r['control_comparison']['passed']:raise ValueError('v78 control codegen changed')
    print((out/'build.log').read_text());print('full-warp metadata compile/audit passed; no runtime claim')


if __name__=='__main__':main()

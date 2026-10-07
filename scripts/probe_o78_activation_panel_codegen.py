#!/usr/bin/env python3
"""v127: one full-K INT32 A-factor panel; no packing or arithmetic changes.

Distinct from register scale hoisting, warp-wide per-stage metadata copies,
and direct global factor loads. Preserve W/payload copies and two-stage MMA.
Gate before GPU execution: >=3% fewer hot instructions, <=168 registers,
no hot local, actual >=3CTAs/SM and identical old control machine code.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_activation_panel_candidate'
STEM='o78_activation_panel'
SHARED=41984
BASELINE=ROOT/'reports/o378_roof_v78_codegen'
DRIVER=ROOT/'reports/o378_roof_v73_codegen'
LIMITS=dict(min_static_reduction=.03,max_allocated_gpr=168,max_hot_local=0,
            required_mma_each=32,required_ldsm=16,required_async_copy=9,
            required_barriers=1,minimum_active_ctas=3)
PRELOAD='''__device__ __forceinline__ void preload_activation_factors(Storage& s,
    const int32_t* af,uint32_t m) {
  // Four16B loads per thread cover exactly32 groups x64 rows. No cast/rounding.
  // These copies join the first payload commit, wait0 and CTA barrier; the
  // panel is never overwritten and subsequent groups need no Af copy.
  o1_static_for<0,4>([&](auto chunk) {
    const unsigned linear=(threadIdx.x+chunk*128)*4;
    const unsigned group=linear/64,row=linear%64;
    copy16(s.activation_factors[group]+row,af+group*m+blockIdx.y*64+row);
  });
}

'''


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('source boundary drift: '+old)
    return text.replace(old,new)


def generated_header(source):
    body=eight_header(source)
    body=once(body,'int activation_factors[2][64];','int activation_factors[32][64];')
    body=once(body,'static_assert(sizeof(Storage)==34304);','static_assert(sizeof(Storage)==41984);')
    body=once(body,'static_assert(sizeof(Storage)==sizeof(C::Storage));',
              '// Extra30 A-factor rows; original payload/weight stage shapes unchanged.')
    body=once(body,'__device__ __forceinline__ void prefetch(',PRELOAD+'__device__ __forceinline__ void prefetch(')
    body=once(body,'  if(threadIdx.x<16) copy16(s.activation_factors[slot]+first,\n'
                    '      af+group*m+blockIdx.y*64+first);\n','')
    body=once(body,'  // Factors share the old scale-panel layout, commit/wait and CTA barrier.',
              '  // Only W factors share the original stage commit/wait and CTA barrier.')
    body=once(body,'  prefetch(s,0,0,a,w,af,wf,m,n,k);',
              '  preload_activation_factors(s,af,m);\n  prefetch(s,0,0,a,w,af,wf,m,n,k);')
    body=once(body,'s.activation_factors[slot][cute::get<0>(coord)]',
              's.activation_factors[group][cute::get<0>(coord)]')
    return body.replace('o78_eight_chain_experiment','o78_activation_panel_experiment')


def mapping(thread,chunk,m,tile_y):
    if not(0<=thread<128 and 0<=chunk<4 and m>0 and m%64==0 and 0<=tile_y<m//64):
        raise ValueError('valid launch coordinates required')
    linear=(thread+chunk*128)*4;group,row=divmod(linear,64)
    return group,row,group*m+tile_y*64+row


def cost_gate(old,new,resources):
    a=next(l for l in old['loops'] if l['kind']=='integer')
    b=next(l for l in new['loops'] if l['kind']=='integer')
    count=lambda p:sum(n for op,n in b['opcode_counts'].items() if op.startswith(p))
    reduction=1-b['static_instructions']/a['static_instructions']
    checks=dict(allocation=new['allocated_gpr']<=168,no_hot_local=count('LDL')+count('STL')==0,
        meaningful_work_reduction=reduction>=LIMITS['min_static_reduction'],
        native_math=count('IMMA.16864.S4.S4')==32 and count('IMMA.16864.U4.S4')==32,
        same_fragments=count('LDSM.')==16,only_weight_metadata_in_hot_copy=count('LDGSTS')==9,
        same_barrier=count('BAR')==1,
        same_residency=resources['control']['active_blocks_per_sm']==resources['candidate']['active_blocks_per_sm']==3)
    return dict(passed=all(checks.values()),checks=checks,limits=LIMITS,
        old_static=a['static_instructions'],new_static=b['static_instructions'],static_reduction=reduction,
        old_allocated_gpr=old['allocated_gpr'],new_allocated_gpr=new['allocated_gpr'],
        scope='predeclared_compile_potential_not_performance_or_MSE')


def query_resources(cubin,library):
    import torch
    torch.cuda.init();anchor=torch.empty(1,device='cuda')
    if torch.cuda.get_device_capability()!=(8,0):raise ValueError('SM80 resource query required')
    lib=ct.CDLL(str(library));lib.roof_probe_error.restype=ct.c_char_p
    lib.roof_probe_open.argtypes=[ct.c_char_p,ct.c_char_p,ct.c_uint,ct.POINTER(ct.c_void_p)]
    lib.roof_probe_resources.argtypes=[ct.c_void_p,ct.POINTER(ct.c_int)]
    lib.roof_probe_close.argtypes=[ct.c_void_p]
    def check(error):
        if error:raise RuntimeError(lib.roof_probe_error().decode())
    rows={}
    for name,symbol,smem in (('control',CONTROL,34304),('candidate',SYMBOL,SHARED)):
        handle=ct.c_void_p();check(lib.roof_probe_open(str(cubin).encode(),symbol.encode(),smem,ct.byref(handle)))
        try:
            values=(ct.c_int*4)();check(lib.roof_probe_resources(handle,values))
            rows[name]=dict(zip(('registers_per_thread','local_bytes','threads','active_blocks_per_sm'),values))
            rows[name]['shared_bytes']=smem
        finally:check(lib.roof_probe_close(handle))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    prior=json.loads((BASELINE/'codegen.json').read_text());driver=json.loads((DRIVER/'build.json').read_text())
    for record in (prior,driver):
        for path,digest in record['sources'].items():
            if sha(ROOT/path)!=digest:raise ValueError('baseline source drift: '+path)
    if sha(BASELINE/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('baseline cubin drift')
    library=DRIVER/'libo78_gpu_prepare.so'
    if sha(library)!=driver['driver_sha256']:raise ValueError('resource driver drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'V12.8.93' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        raise ValueError('pinned CUDA12.8.93/CUTLASS required')
    out.mkdir(parents=True)
    source=(ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out/'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    (out/(STEM+'_generated.cuh')).write_text(generated_header(source))
    names=set(prior['sources'])|{'csrc/sm80/roof_o78_activation_panel_probe.cu',
        'scripts/probe_o78_activation_panel_codegen.py','scripts/inspect_eight_chain_schedule.py',
        'scripts/compare_a100_codegen.py','csrc/sm80/roof_producer_warp_driver.cpp'}
    receipt=dict(scope='v127_allK_activation_factor_panel_compile_not_GPU_validation',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(names)},baseline_cubin_sha256=prior['cubin_sha256'],
        resource_driver_sha256=sha(library),nvcc=version,cutlass_commit=commit,limits=LIMITS,
        production_default_changed=False,math_and_quantization_unchanged=True,commands=[],
        cta_tile=[64,128,128],threads=128,stages=2,shared_bytes=SHARED,
        activation_panel_dtype='int32_no_compression',activation_panel_shape=[32,64],
        copies_join_first_commit_wait0_barrier=True,no_candidate_kernel_launched=True)
    def run(cmd,name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    src=str(ROOT/'csrc/sm80/roof_o78_activation_panel_probe.cu');cubin=out/(STEM+'.cubin')
    run(flags+[src,'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[src,'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();ptx=(out/(STEM+'.ptx')).read_text()
    es=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    live={s:analyze((out/'liveness.txt').read_text(),s) for s in (CONTROL,SYMBOL)}
    for s in (CONTROL,SYMBOL):
        entry=next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in entry for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX native math/copy audit failed')
    schedules={s:trace(sass,s,live[s]) for s in (CONTROL,SYMBOL)}
    resources=query_resources(cubin,library)
    control=compare((BASELINE/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    gate=cost_gate(live[CONTROL],live[SYMBOL],resources)
    gate['checks']['control_encoding_unchanged']=control['passed'];gate['passed']=all(gate['checks'].values())
    receipt.update(entries=es,liveness=live,schedules=schedules,runtime_resources=resources,
        cost_gate=gate,control_comparison=control,cubin_sha256=sha(cubin),
        artifact_sha256={f.name:sha(f) for f in out.iterdir() if f.is_file()})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if not control['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4'] and
            not e['int8_mma'] and e['all_copies_bypass_l1'] for e in es.values()):
        raise ValueError('native math/control audit failed')
    print(json.dumps(dict(gate=gate,resources=resources),indent=2),flush=True)


if __name__=='__main__':main()

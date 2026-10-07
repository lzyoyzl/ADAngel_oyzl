#!/usr/bin/env python3
"""v115 fixed O3 DP2A mainloop, guarded per N128, not tile/chain enumeration.

Only proceed after the separate scalar ISA and full24 data gate passes.
Keep v89 default/control, 64x128x128,128threads,3stage and G128 semantics.
Do not launch if resources, local traffic, native INT4 or dot lowering fail.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from compare_a100_codegen import compare
from probe_grouped_cta_codegen import ROOT,checked as baseline_checked
from probe_roof_fullk_integer_codegen import static_entries

SYMBOL='adangel_roof_o3_dp2a_candidate'
CONTROL='adangel_roof_o3_grouped_cta_candidate'
BASELINE=Path('reports/o378_roof_v89_o3_codegen')
PRIOR=Path('reports/o378_roof_v115_dp2a_cost')
STEM='o3_dp2a'
BODY='''      // Eight separate low/high chains, each two K64 MMA steps.
      //32 partial registers, not the previous eight merged four-step chains.
      o1_static_for<0,2>([&](auto mi) {
        auto pl=cute::make_tensor<int>(cute::make_shape(cute::_4{},cute::_4{}));
        auto ph=cute::make_fragment_like(pl);cute::clear(pl);cute::clear(ph);
        o1_static_for<0,4>([&](auto ni) {
          auto l=pl(cute::_,ni),h=ph(cute::_,ni);
          cute::gemm(LA{},l,a00(cute::_,mi,cute::_0{}),b00(cute::_,ni,cute::_0{}),l);
          cute::gemm(HA{},h,h00(cute::_,mi,cute::_0{}),b00(cute::_,ni,cute::_0{}),h);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto l=pl(cute::_,ni),h=ph(cute::_,ni);
          cute::gemm(LA{},l,a01(cute::_,mi,cute::_0{}),b01(cute::_,ni,cute::_0{}),l);
          cute::gemm(HA{},h,h01(cute::_,mi,cute::_0{}),b01(cute::_,ni,cute::_0{}),h);
        });
        o1_static_for<0,4>([&](auto ni) {
          auto full_ni=nb*cute::_4{}+ni;
          o1_static_for<0,4>([&](auto vi) {
            const int col=cute::get<1>(coords(vi,mi,full_ni));
            uint32_t dots;
            asm("prmt.b32 %0, %1, %2, 0x5410;"
                : "=r"(dots) : "r"(pl(vi,ni)), "r"(ph(vi,ni)));
            int value;
            asm("dp2a.lo.s32.u32 %0, %1, %2, %3;"
                : "=r"(value) : "r"(dots), "r"(s.factor[slot][col]), "r"(acc(vi,mi,full_ni)));
            acc(vi,mi,full_ni)=value;
          });
        });
      });
'''


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def generated_header(source):
    begin='      // Eight independent chains: two M atoms x four N atoms.'
    end='    });\n  }\n  auto final_value='
    if source.count(begin)!=1 or source.count(end)!=1:raise ValueError('v89 eight-chain boundary drift')
    text=source[:source.index(begin)]+BODY+source[source.index(end):]
    text=text.replace('o3_grouped_cta_experiment','o3_dp2a_experiment')
    old='factors+group*n+roof_grouped_cta::tile().x*128+first'
    if text.count(old)!=1:raise ValueError('factor prefetch boundary drift')
    # Epilogue anchors still occupy original row32, not packed-panel row64.
    return text.replace(old,'factors+(33+group)*n+roof_grouped_cta::tile().x*128+first')


def pack_metadata_reference(original):
    x=np.asarray(original)
    if x.dtype!=np.int32 or x.ndim!=2 or x.shape[0]!=33 or x.shape[1]%128:
        raise ValueError('INT32[33,N128] metadata required')
    n=x.shape[1];fits=(x[:32]>=1)&(x[:32]<=15)
    packed=np.where(fits,x[:32].astype(np.int64)*4097,0).astype(np.int32)
    flags=fits.reshape(32,n//128,128).all(axis=(0,2)).astype(np.int32)
    return np.concatenate((x.ravel(),packed.ravel(),flags))


def dot_liveness(text,symbol=SYMBOL):
    blocks=re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
    block=next((x for x in blocks if re.match(r'//-+ \.text\.'+re.escape(symbol)+r'\s',x)),None)
    if block is None:raise ValueError('exact dot entry liveness missing')
    registers=re.search(r'SHI_REGISTERS=(\d+)',block)
    if registers is None:raise ValueError('allocation metadata missing')
    ops=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_\w+):',line)
        if label:pending.append(label[1])
        ins=re.search(r'/\*([0-9a-f]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if not ins:continue
        pc=int(ins[1],16);op=re.sub(r'^@!?\w+\s+','',ins[2].strip())
        for label in pending:labels[label]=pc
        pending.clear();ops.append(dict(pc=pc,instruction=op,live=int(ins[3])))
    matches=[]
    for ins in ops:
        branch=re.search(r'\bBRA\s+`\((\.L_\w+)\)',ins['instruction'])
        if not branch:continue
        if branch[1] not in labels:raise ValueError('branch target missing')
        begin=labels[branch[1]]
        if begin>=ins['pc']:continue
        region=[x for x in ops if begin<=x['pc']<=ins['pc']]
        counts=Counter(x['instruction'].split()[0] for x in region)
        if counts.get('IDP.2A.LO.S16.U8')!=64:continue
        matches.append(dict(begin_pc=hex(begin),end_pc=hex(ins['pc']),
            static_instructions=len(region),max_live_gpr=max(x['live'] for x in region),
            opcode_counts=dict(sorted(counts.items()))))
    if len(matches)!=1:raise ValueError('one exact64-DP2A integer G128 loop required')
    return dict(allocated_gpr=int(registers[1]),function_max_live_gpr=max(x['live'] for x in ops),loop=matches[0])


def gate(old,new):
    a=next(x for x in old['loops'] if x['kind']=='integer');b=new['loop'];ops=b['opcode_counts']
    local=lambda c:sum(v for k,v in c.items() if k.split('.')[0] in ('LDL','STL'))
    r=dict(registers_at_most168=new['allocated_gpr']<=168,
        hot_local_not_worse=local(ops)<=local(a['opcode_counts']),
        native_dot64=ops.get('IDP.2A.LO.S16.U8')==64,
        native_int4_unchanged=ops.get('IMMA.16864.U4.S4')==ops.get('IMMA.16864.S4.S4')==32,
        ldsm_unchanged=ops.get('LDSM.16.M88.4')==16,
        no_hot_i2f=not ops.get('I2F'),
        static_work_ratio=b['static_instructions']/a['static_instructions'])
    r['work_growth_at_most5pct']=r['static_work_ratio']<=1.05
    r['passed']=all(v for k,v in r.items() if k!='static_work_ratio')
    r['scope']='structural cost gate, not hardware latency/performance/correctness proof'
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh project output required')
    previous=json.loads((ROOT/PRIOR/'analysis.json').read_text())
    if not previous['proceed_with_O3_GEMM']:raise ValueError('scalar/data gate failed')
    for path,sha in previous['source_hashes'].items():
        if digest(ROOT/path)!=sha:raise ValueError('ISA source drift')
    for path,sha in previous['artifact_sha256'].items():
        if digest(ROOT/PRIOR/path)!=sha:raise ValueError('ISA artifact drift')
    baseline=ROOT/BASELINE;baseline_checked(baseline,'o3')
    prior=json.loads((baseline/'codegen.json').read_text())
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    cutlass_sha=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'V12.8.93' not in version or cutlass_sha!='db1c288993354c88e551c40c19a8fb93a774a241':raise ValueError('pinned toolchain required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu','.cuh'):(out/f.name).write_text(f.read_text())
    (out/'o3_dp2a_generated.cuh').write_text(generated_header((baseline/'o3_grouped_cta_generated.cuh').read_text()))
    sources=set(prior['sources'])|{'csrc/sm80/roof_o3_dp2a_probe.cu','scripts/probe_o3_dp2a_codegen.py'}
    r=dict(scope='v115_full_O3_DP2A_compile_gate_not_GPU_timing_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={s:digest(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=cutlass_sha,
        prior_ISA_sha256=digest(ROOT/PRIOR/'analysis.json'),baseline_cubin_sha256=prior['cubin_sha256'],
        symbol=SYMBOL,cta_tile=[64,128,128],threads=128,stages=3,shared_bytes=50688,
        partial_registers=32,final_accumulator_registers=64,scalar_recomposition='exact_PRMT_plus_DP2A',
        original_G128_scales_unchanged=True,metadata_layout='original33*N + packed32*N + flagsN128',
        metadata_cost_must_count_in_Cold=True,production_default_changed=False,candidate_GPU_launched=False,
        new_performance_result=False,new_MSE_result=False)
    extension=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extension)!=1:raise ValueError('one formal extension required')
    r['formal_extension_sha256_before']=digest(extension[0])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-arch=sm_80','-lineinfo',
        '-DADANGEL_FULLK_INTEGER=1','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/'csrc/sm80/roof_o3_dp2a_probe.cu')]
    cubin=out/'o3_dp2a.cubin'
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/'o3_dp2a.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o3_dp2a.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/'o3_dp2a.sass').read_text();symbols={SYMBOL,CONTROL}
    e=static_entries(sass,'^(?:'+'|'.join(symbols)+')$',symbols)
    if not all(x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma'] and x['all_copies_bypass_l1'] for x in e.values()):raise ValueError('same-entry INT4/copy audit failed')
    comparison=compare((baseline/'o3_grouped_cta.sass').read_text(),sass,'^'+CONTROL+'$')
    if not comparison['passed']:raise ValueError('unchanged v89 control encoding drift')
    live=dot_liveness((out/'liveness.txt').read_text())
    r.update(entries=e,liveness=live,control_comparison=comparison,compile_gate=gate(prior['liveness'][CONTROL],live),
        cubin_sha256=digest(cubin),formal_extension_sha256_after=digest(extension[0]))
    if r['formal_extension_sha256_after']!=r['formal_extension_sha256_before']:raise ValueError('formal extension changed')
    r['artifact_sha256']={f.name:digest(f) for f in out.iterdir() if f.is_file()}
    (out/'analysis.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gate=r['compile_gate'],liveness=live,candidate_GPU_launched=False),indent=2),flush=True)


if __name__=='__main__':main()

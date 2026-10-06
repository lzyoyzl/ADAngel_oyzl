#!/usr/bin/env python3
"""v111: one128x192/12warp CTA, constant per-warp math/register budget.

Unlike v57/v83 (more output registers/thread) and v98 (more LDSM/output),
preserve64 acc/32 partial/eight chains/16 LDSM per warp. Intended one12-warp
CTA/SM versus three4-warp CTAs. Include3.125% padding work and boundary checks.
Compiler/host-coordinate gates only. General fallback is NOT integrated yet;
no GPU candidate output/performance/MSE until this gate and that work pass.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_o78_eight_chain_codegen import ROOT
from probe_roof_fullk_integer_codegen import static_entries

BASELINE=ROOT/'reports/o378_roof_v78_codegen'
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOL='adangel_roof_o78_cooperative_reuse_candidate'
STEM='o78_cooperative_reuse'
SHARED=59904

CONFIG='''// v111: unchanged math, proportional CTA cooperation, fixed candidate.
#pragma once
namespace o78_cooperative_reuse_experiment {
struct C {
  using WarpLayout=cute::Layout<cute::Shape<cute::_4,cute::_3,cute::_1>>;
  using Mma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      WarpLayout,cute::Tile<cute::_128,cute::_192,cute::_64>>;
  using HighMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32S4S4S32_TN>,
      WarpLayout,cute::Tile<cute::_128,cute::_192,cute::_64>>;
  using SliceMma=cute::TiledMMA<cute::MMA_Atom<cute::SM80_16x8x64_S32U4S4S32_TN>,
      WarpLayout,cute::Tile<cute::_128,cute::_96,cute::_64>>;
  template<int Rows> using ByteLayout=decltype(cute::composition(
      cute::Swizzle<2,4,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::_64>,
      cute::Stride<cute::_64,cute::_1>>{}));
  template<int Rows> using NibbleLayout=decltype(cute::composition(
      cute::Swizzle<2,5,3>{},cute::Layout<cute::Shape<cute::Int<Rows>,cute::_128>,
      cute::Stride<cute::_128,cute::_1>>{}));
};
struct alignas(128) Storage {
  alignas(16) int activation_factors[2][128];
  alignas(128) uint8_t low[2][128*64],high[2][128*64],weight[2][192*64];
  int weight_factors[2][192];
};
static_assert(sizeof(Storage)==59904);

__device__ __forceinline__ void guarded_copy16(void* dst,const void* src,bool valid) {
  const uint32_t address=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  // Documented ignore-src predicate zero-fills padding. Keep src in bounds.
  asm volatile("{ .reg .pred ignore; setp.eq.u32 ignore, %2, 0; "
      "cp.async.cg.shared.global [%0], [%1], 16, ignore; }" ::
      "r"(address),"l"(src),"r"(unsigned(valid)):"memory");
}
__device__ __forceinline__ void prefetch(Storage& s,int slot,int group,
    const uint8_t* a,const uint8_t* w,const int32_t* af,const int32_t* wf,
    uint32_t m,uint32_t n,uint32_t k) {
  C::ByteLayout<128> la;C::ByteLayout<192> lb;
  o1_static_for<0,2>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*6144;
    if(off<128*64) {
      unsigned row=off/64,col=off%64;
      const auto* src=a+group*m*64+(blockIdx.y*128+row)*64+col;
      copy16(s.low[slot]+la(row,col),src);
      copy16(s.high[slot]+la(row,col),src+m*(k/2));
    }
  });
  o1_static_for<0,2>([&](auto chunk) {
    unsigned off=threadIdx.x*16+chunk*6144,row=off/64,col=off%64;
    unsigned column=blockIdx.x*192+row;
    unsigned safe_column=column<n?column:0u;
    guarded_copy16(s.weight[slot]+lb(row,col),w+group*n*64+safe_column*64+col,column<n);
  });
  const unsigned first=threadIdx.x*4;
  if(threadIdx.x<32)copy16(s.activation_factors[slot]+first,
      af+group*m+blockIdx.y*128+first);
  if(threadIdx.x<48) {
    const unsigned column=blockIdx.x*192+first;
    guarded_copy16(s.weight_factors[slot]+first,
        wf+group*n+(column<n?column:0u),column<n);
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

'''


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('v78 source boundary drift: '+old[:70])
    return source.replace(old,new)


def generated_header(source):
    body=source[source.index('__device__ __forceinline__ void body('):]
    body=body.replace('o78_eight_chain_experiment','o78_cooperative_reuse_experiment')
    body=once(body,'cute::make_shape(cute::_64{},cute::_128{})',
              'cute::make_shape(cute::_128{},cute::_192{})')
    if body.count('C::NibbleLayout<64>{}')!=2:raise ValueError('two A layouts required')
    body=body.replace('C::NibbleLayout<128>{}','C::NibbleLayout<192>{}')
    body=body.replace('C::NibbleLayout<64>{}','C::NibbleLayout<128>{}')
    body=once(body,'cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(cute::_0{},half)',
        'cute::make_shape(cute::_128{},cute::_64{}),cute::make_coord(cute::_0{},half)')
    body=once(body,'using SliceMma=O78::O3AmpereConfig<64,64,128,false,2>::Mma;',
        'using SliceMma=C::SliceMma;')
    body=once(body,'cute::make_shape(cute::_64{},cute::_64{}),cute::make_coord(nb,half)',
        'cute::make_shape(cute::_96{},cute::_64{}),cute::make_coord(nb,half)')
    body=body.replace('blockIdx.y*64','blockIdx.y*128').replace('blockIdx.x*128','blockIdx.x*192')
    body=once(body,'const float column=base_w[blockIdx.x*192+cute::get<1>(p)];',
        'const unsigned global_column=blockIdx.x*192+cute::get<1>(p);\n'
        '    const float column=global_column<n?base_w[global_column]:0.0f;')
    body=once(body,'    if(cute::get<0>(p)==cute::get<0>(q)',
        '    if(blockIdx.x*192+cute::get<1>(p)>=n)return; //Compile-time pair lambda only.\n'
        '    if(cute::get<0>(p)==cute::get<0>(q)')
    return CONFIG+body


def work_model():
    old_tiles=64*32;new_tiles=32*22
    old_bytes=old_tiles*(64*128+128*64+(64+128)*4)
    # The final tile has64 valid columns; zero-fill its128 padding columns.
    new_bytes=32*(22*(128*128+128*4)+4096*(64+4))
    old_mma=old_tiles*4*64;new_mma=new_tiles*12*64
    return dict(shape=[4096,4096,4096],old_ctas=old_tiles,new_ctas=new_tiles,
        old_warps_per_cta=4,new_warps_per_cta=12,old_resident_ctas=3,intended_resident_ctas=1,
        old_resident_warps=12,intended_resident_warps=12,acc_per_thread=64,partial_per_thread=32,
        padded_mma_ratio=new_mma/old_mma,old_input_bytes_per_group=old_bytes,
        new_input_bytes_per_group=new_bytes,input_bytes_ratio=new_bytes/old_bytes,
        interpretation='logical_work_model_not_speedup_or_physical_DRAM_traffic')


def analyze(text,symbol):
    block=next(b for b in re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
               if re.match(r'//-+ \.text\.'+re.escape(symbol)+r'\s',b))
    allocated=int(re.search(r'SHI_REGISTERS=(\d+)',block)[1]);ins=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label:pending.append(label[1])
        match=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if match:
            pc=int(match[1],16)
            for lab in pending:labels[lab]=pc
            pending=[];ins.append((pc,match[2].strip(),int(match[3])))
    loops=[]
    for pc,op,_ in ins:
        target=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',op)
        if not target or labels.get(target[1],pc)>=pc:continue
        region=[i for i in ins if labels[target[1]]<=i[0]<=pc]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',i[1]).split()[0] for i in region)
        if sum(v for k,v in counts.items() if k.startswith('IMMA.'))!=64:continue
        if counts.get('I2F',0):continue
        loops.append(dict(begin_pc=hex(region[0][0]),end_pc=hex(pc),static_instructions=len(region),
            max_live_gpr=max(i[2] for i in region),opcode_counts=dict(sorted(counts.items()))))
    if len(loops)!=1:raise ValueError('one full64-MMA integer loop required')
    return dict(symbol=symbol,allocated_gpr=allocated,loop=loops[0])


def gate(old,new):
    ops=new['loop']['opcode_counts'];model=work_model()
    ratio=model['padded_mma_ratio']*new['loop']['static_instructions']/old['loop']['static_instructions']
    checks=dict(registers_at_most168=new['allocated_gpr']<=168,
        native_mma64=ops.get('IMMA.16864.S4.S4')==ops.get('IMMA.16864.U4.S4')==32,
        no_fragment_read_growth=ops.get('LDSM.16.M88.4')==16,
        hot_local_at_most2=sum(v for k,v in ops.items() if k.startswith(('LDL','STL')))<=2,
        weighted_instructions_at_least5pct_less=ratio<=.95)
    return dict(checks=checks,passed=all(checks.values()),weighted_static_instruction_ratio=ratio,
        runtime_residency_gate='must query1 resident CTA and12 warps before any runtime test',
        acceptance_incomplete='fallback/runtime/MSE/24samples not established by compile gate')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    prior=json.loads((BASELINE/'codegen.json').read_text())
    for path,digest in prior['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('v78 source drift: '+path)
    if sha(BASELINE/'o78_eight_chain.cubin')!=prior['cubin_sha256']:raise ValueError('v78 binary drift')
    cuda=Path('/usr/local/cuda-12.8/bin');cutlass=ROOT/'third_party/cutlass-src'
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    commit=subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit!='db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    (out/'o78_eight_chain_generated.cuh').write_text((BASELINE/'o78_eight_chain_generated.cuh').read_text())
    (out/(STEM+'_generated.cuh')).write_text(generated_header((BASELINE/'o78_eight_chain_generated.cuh').read_text()))
    sources=set(prior['sources'])|{'scripts/probe_cooperative_reuse_codegen.py',
        'csrc/sm80/roof_o78_cooperative_reuse_probe.cu','tests/cuda/validate_cooperative_reuse_coordinates.cu'}
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one production extension required')
    r=dict(scope='v111_one_proportional_CTA_compile_prototype_not_runtime_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},nvcc=version,cutlass_commit=commit,commands=[],
        cta_tile=[128,192,128],threads=384,stages=2,shared_bytes=SHARED,work_model=work_model(),
        per_group_scale_semantics_changed=False,production_default_changed=False,
        native_extension_rebuilt=False,extension_sha256_before=sha(extensions[0]),
        full_fallback_integration_pending=True,candidate_gpu_launched=False,
        new_real_GEMM_measured=False,new_MSE_measured=False,baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    common=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    coordinate=out/'validate_coordinates'
    run(common+['-O2',str(ROOT/'tests/cuda/validate_cooperative_reuse_coordinates.cu'),'-o',str(coordinate)],'coordinates_build.log')
    run([str(coordinate)],'coordinates.log');r['coordinates']=json.loads((out/'coordinates.log').read_text())
    if not r['coordinates']['passed'] or r['coordinates']['gpu_execution']:raise ValueError('host mapping failed')
    flags=common+['-O3','-lineinfo',str(ROOT/'csrc/sm80/roof_o78_cooperative_reuse_probe.cu')]
    cubin=out/(STEM+'.cubin')
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();live=(out/'liveness.txt').read_text()
    r['control_comparison']=compare((BASELINE/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    if not r['control_comparison']['passed']:raise ValueError('old control code changed')
    r['entries']=static_entries(sass,'^(?:'+CONTROL+'|'+SYMBOL+')$',{CONTROL,SYMBOL})
    for e in r['entries'].values():
        if not e['native_s4_s4'] or not e['native_u4_s4'] or e['int8_mma']:
            raise ValueError('same-entry native INT4 audit failed')
    r['liveness']={s:analyze(live,s) for s in (CONTROL,SYMBOL)}
    r['compile_gate']=gate(r['liveness'][CONTROL],r['liveness'][SYMBOL])
    r['extension_sha256_after']=sha(extensions[0])
    if r['extension_sha256_after']!=r['extension_sha256_before']:raise ValueError('extension changed')
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='codegen.json'}
    (out/'codegen.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(compile_gate=r['compile_gate'],liveness=r['liveness'],coordinates=r['coordinates']),indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""v110: matched eight-chain/64-output capacity shells, not a GEMM candidate.

New diagnostic beyond v82/v102 (MMA only): retain actual CuTe operand layout,
all64 integer output accumulators, then add recurring LDSM and G128 factors.
No quantizer/default/native-extension changes; no synthetic result is a real
trace latency or a guaranteed achievable kernel peak. Three CTAs/SM required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_roof_fullk_integer_codegen import static_entries

ROOT=Path(__file__).resolve().parents[1]
BASELINE=ROOT/'reports/o378_roof_v78_codegen'
STEM='o78_shell_capacity'
CONTROL='adangel_roof_o78_eight_chain_candidate'
SYMBOLS=('adangel_capacity_register_shell','adangel_capacity_shared_shell',
         'adangel_capacity_scaled_shared_shell')
SHARED=50688  # Deliberate occupancy reservation, not a production SMEM claim.

STORAGE='''struct alignas(128) Storage {
  alignas(16) int activation_factors[32][64];
  alignas(128) uint8_t low[1][64*64],high[1][64*64],weight[1][128*64];
  int weight_factors[32][128];
};
static_assert(sizeof(Storage)==40960);

template<bool Scale>
__device__ __forceinline__ void initialize(Storage& s,int seed) {
  C::ByteLayout<64> la;C::ByteLayout<128> lb;
  for(int off=threadIdx.x;off<64*64;off+=128) {
    int row=off/64,col=off%64;
    int value=(int(blockIdx.y)*64+row+seed)%5-2;
    s.low[0][la(row,col)]=uint8_t((value&15)*17);
    s.high[0][la(row,col)]=uint8_t(((value<0)?15:0)*17);
  }
  for(int off=threadIdx.x;off<128*64;off+=128) {
    int row=off/64,col=off%64;
    int value=(int(blockIdx.x)*128+row+seed)%7-3;
    s.weight[0][lb(row,col)]=uint8_t((value&15)*17);
  }
  if constexpr(Scale) {
    for(int off=threadIdx.x;off<32*64;off+=128) {
      int g=off/64,row=off%64;
      s.activation_factors[g][row]=1+(int(blockIdx.y)*64+row+g+seed)%3;
    }
    for(int off=threadIdx.x;off<32*128;off+=128) {
      int g=off/128,col=off%128;
      s.weight_factors[g][col]=1+(int(blockIdx.x)*128+col+2*g+seed)%3;
    }
  }
  __syncthreads(); // Read-only shared data afterwards; no unsafe slot reuse.
}

'''


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('v78 boundary drift: '+old[:90])
    return source.replace(old,new)


def generated_header(source):
    begin=source.index('struct alignas(128) Storage {')
    end=source.index('__device__ __forceinline__ void body(')
    source=source[:begin]+STORAGE+source[end:]
    source=once(source,'''__device__ __forceinline__ void body(const uint8_t* a,const uint8_t* w,
    const int32_t* af,const int32_t* wf,const float* base_a,const float* base_w,
    float* y,uint32_t m,uint32_t n,uint32_t k) {''',
        '''template<int Mode>
__device__ __forceinline__ void body(float* y,int groups,int seed) {
  static_assert(Mode>=0 && Mode<=2);
  constexpr uint32_t m=4096,n=4096,k=4096;''')
    source=once(source,'''  constexpr int Groups=32;
  prefetch(s,0,0,a,w,af,wf,m,n,k);
  for(int group=0;group<Groups;++group) {
    const int slot=group%2;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads(); // All prior readers finish before a slot is reused.
    if(group+1<Groups) prefetch(s,1-slot,group+1,a,w,af,wf,m,n,k);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(slot),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(slot),cute::_1{})),hd1);''',
        '''  auto b20=cute::make_fragment_like(b0),b21=cute::make_fragment_like(b1);
  auto load_a=[&]() {
    cute::copy(LCopy{},lc.partition_S(tile_a(low(0),cute::_0{})),ld0);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(0),cute::_0{})),hd0);
    cute::copy(LCopy{},lc.partition_S(tile_a(low(0),cute::_1{})),ld1);
    cute::copy(SCopy{},hc.partition_S(tile_a(high(0),cute::_1{})),hd1);
  };
  initialize<Mode==2>(s,seed);
  if constexpr(Mode==0) {
    load_a();
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_0{},cute::_0{})),bd0);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_0{},cute::_1{})),bd1);
    auto bd20=bc.retile_D(b20),bd21=bc.retile_D(b21);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_1{},cute::_0{})),bd20);
    cute::copy(SCopy{},bc.partition_S(tile_b(0,cute::_1{},cute::_1{})),bd21);
  }
  #pragma unroll 1
  for(int group=0;group<groups;++group) {
    constexpr int slot=0;
    if constexpr(Mode!=0) load_a();''')
    source=once(source,'''      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
      cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);''',
        '''      if constexpr(Mode!=0) {
        cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_0{})),bd0);
        cute::copy(SCopy{},bc.partition_S(tile_b(slot,nb,cute::_1{})),bd1);
      }
      auto use_b0=[&]() {
        if constexpr(Mode==0 && decltype(nb)::value==1) return b20(cute::_,cute::_,cute::_);
        else return b0(cute::_,cute::_,cute::_);
      }();
      auto use_b1=[&]() {
        if constexpr(Mode==0 && decltype(nb)::value==1) return b21(cute::_,cute::_,cute::_);
        else return b1(cute::_,cute::_,cute::_);
      }();''')
    for name in ('b0','b1'):
        source=source.replace(name+'(cute::_,ni,cute::_0{})','use_'+name+'(cute::_,ni,cute::_0{})')
    source=once(source,'''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];
            acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;''',
        '''            if constexpr(Mode==2) {
              const int coefficient=s.activation_factors[group&31][cute::get<0>(coord)]*
                                    s.weight_factors[group&31][cute::get<1>(coord)];
              acc(vi,mi,full_ni)+=partial(vi,mi,ni)*coefficient;
            } else acc(vi,mi,full_ni)+=partial(vi,mi,ni);''')
    source=once(source,'''    const float row=base_a[blockIdx.y*64+cute::get<0>(p)];
    const float column=base_w[blockIdx.x*128+cute::get<1>(p)];
    acc(i)=__float_as_int(__fmul_rn(__fmul_rn(float(acc(i)),row),column));''',
        '''    // Synthetic |integer|<=256*128*2*3*9=1769472, exactly FP32.
    acc(i)=__float_as_int(float(acc(i)));''')
    return source.replace('o78_eight_chain_experiment','o78_shell_capacity_experiment')


def hot_loop(text,symbol):
    block=next(b for b in re.split(r'(?=^//-+ \.text\.)',text,flags=re.M)
               if re.match(r'//-+ \.text\.'+re.escape(symbol)+r'\s',b))
    allocated=int(re.search(r'SHI_REGISTERS=(\d+)',block)[1])
    ins=[];labels={};pending=[]
    for line in block.splitlines():
        label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
        if label:pending.append(label[1])
        match=re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|',line)
        if match:
            pc=int(match[1],16)
            for name in pending:labels[name]=pc
            pending=[];ins.append((pc,match[2].strip(),int(match[3])))
    candidates=[]
    for pc,op,live in ins:
        branch=re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)',op)
        if not branch or labels.get(branch[1],pc)>=pc:continue
        region=[x for x in ins if labels[branch[1]]<=x[0]<=pc]
        counts=Counter(re.sub(r'^@!?[A-Z0-9]+\s+','',x[1]).split()[0] for x in region)
        if sum(v for k,v in counts.items() if k.startswith('IMMA.'))==64:
            candidates.append(dict(static_instructions=len(region),opcode_counts=dict(sorted(counts.items())),
                max_live_gpr=max(x[2] for x in region),begin_pc=hex(region[0][0]),end_pc=hex(pc)))
    if len(candidates)!=1:raise ValueError('one64-MMA hot loop expected: '+symbol)
    return dict(symbol=symbol,allocated_gpr=allocated,loop=candidates[0])


def compile_gate(rows):
    checks=[]
    for mode,symbol in enumerate(SYMBOLS):
        row=rows[symbol];ops=row['loop']['opcode_counts']
        checks.append(dict(symbol=symbol,
            native_two_INT4_routes=ops.get('IMMA.16864.S4.S4')==ops.get('IMMA.16864.U4.S4')==32,
            expected_LDSM=sum(v for k,v in ops.items() if k.startswith('LDSM.'))==(0 if mode==0 else 16),
            no_hot_local=not any(v for k,v in ops.items() if k.startswith(('LDL','STL'))),
            no_hot_barrier=not any(v for k,v in ops.items() if k.startswith(('BAR.','WARPSYNC'))),
            no_hot_global_copy=not any(v for k,v in ops.items() if k.startswith('LDGSTS')),
            allocated_at_most168=row['allocated_gpr']<=168))
    return dict(passed=all(v for check in checks for k,v in check.items() if k!='symbol'),checks=checks,
        runtime_gate='CUDA query must confirm3 resident CTAs/SM; checksum before timings')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--run',action='store_true',help='run diagnostics only after compiler gate')
    args=p.parse_args();out=args.output.resolve()
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
    (out/'o78_shell_generated.cuh').write_text(generated_header((BASELINE/'o78_eight_chain_generated.cuh').read_text()))
    sources=set(prior['sources'])|{'scripts/probe_o78_shell_capacity.py',
        'csrc/sm80/roof_o78_shell_probe.cu','csrc/sm80/roof_o78_shell_capacity_driver.cpp'}
    extensions=list((ROOT/'python/adangel').glob('_sm80*.so'))
    if len(extensions)!=1:raise ValueError('one unchanged production extension required')
    r=dict(scope='v110_fixed_grid_capacity_diagnostic_not_real_trace_GEMM',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(sources)},commands=[],nvcc=version,cutlass_commit=commit,
        cta_tile=[64,128,128],grid=[32,64],threads=128,stages_in_hot_loop=0,
        independent_chains=8,partial_registers=32,output_accumulators=64,
        shared_used_bytes=40960,shared_reserved_bytes=SHARED,groups_for_timing=256,
        initialization_and_output_not_removed_from_event_time=True,
        normalization_to32_groups_not_actual_4096_trace_latency=True,
        new_MSE_measured=False,new_real_GEMM_measured=False,production_default_changed=False,
        extension_sha256_before=sha(extensions[0]),baseline_cubin_sha256=prior['cubin_sha256'])
    def run(cmd,name):
        r['commands'].append(cmd)
        with (out/name).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,TMPDIR=str(ROOT/'tmp')),
                stdout=log,stderr=subprocess.STDOUT,check=True)
    flags=[str(cuda/'nvcc'),'-std=c++17','--expt-relaxed-constexpr','-O3','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out)]
    source=ROOT/'csrc/sm80/roof_o78_shell_probe.cu';cubin=out/(STEM+'.cubin')
    run(flags+[str(source),'-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+[str(source),'-ptx','-o',str(out/(STEM+'.ptx'))],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],STEM+'.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass=(out/(STEM+'.sass')).read_text();live=(out/'liveness.txt').read_text()
    r['liveness']={s:hot_loop(live,s) for s in SYMBOLS}
    r['control_comparison']=compare((BASELINE/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$')
    if not r['control_comparison']['passed']:raise ValueError('old control code changed')
    r['entries']=static_entries(sass,'^(?:'+'|'.join(SYMBOLS)+')$',set(SYMBOLS))
    for e in r['entries'].values():
        if not e['native_u4_s4'] or not e['native_s4_s4'] or e['int8_mma']:
            raise ValueError('same-entry native INT4 audit failed')
    r['compile_gate']=compile_gate(r['liveness'])
    if args.run and r['compile_gate']['passed']:
        executable=out/'capacity_driver'
        run([str(cuda/'nvcc'),'-std=c++17','-O2',str(ROOT/'csrc/sm80/roof_o78_shell_capacity_driver.cpp'),
            '-lcuda','-o',str(executable)],'driver_build.log')
        run(['nvidia-smi','--query-gpu=name,clocks.sm,temperature.gpu,power.draw','--format=csv'],'gpu_before.txt')
        run([str(executable),str(cubin)],'results.jsonl')
        raw=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
        if len(raw)!=9 or {(x['round'],x['mode']) for x in raw}!={(r,m) for r in range(3) for m in range(3)}:
            raise ValueError('three rounds/three modes required')
        import numpy as np
        stats=[]
        for row in raw:
            x=np.array(row['raw_ms'],dtype=np.float64)
            if len(x)!=200 or not np.isfinite(x).all() or np.any(x<=0):raise ValueError('bad Event timing')
            stats.append(dict(mode=row['mode'],round=row['round'],median_ms=float(np.median(x)),
                normalized32_groups_ms=float(np.median(x)/8),mean_ms=float(np.mean(x)),
                cv_percent=float(np.std(x)/np.mean(x)*100),p5_ms=float(np.quantile(x,.05)),p95_ms=float(np.quantile(x,.95))))
        r['runtime']=dict(records=9,checks_before_timing=raw[0]['validation_checks'],
            checksum_passed=all(x['checksum_passed'] for x in raw),
            active_ctas_per_sm=[raw[i]['active_ctas_per_sm'] for i in range(3)],statistics=stats)
        run(['nvidia-smi','--query-gpu=name,clocks.sm,temperature.gpu,power.draw','--format=csv'],'gpu_after.txt')
    r['extension_sha256_after']=sha(extensions[0])
    if r['extension_sha256_after']!=r['extension_sha256_before']:raise ValueError('production binary changed')
    r['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='analysis.json'}
    (out/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(compile_gate=r['compile_gate'],liveness=r['liveness'],runtime=r.get('runtime')),indent=2),flush=True)


if __name__=='__main__':main()

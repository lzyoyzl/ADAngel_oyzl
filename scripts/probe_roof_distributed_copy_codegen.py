#!/usr/bin/env python3
"""v64: distribute next-stage copies among independent current-stage MMAs.

Scheduling inspiration (not copied implementation): pinned CUTLASS SM80
mma_multistage.h, mac_loop_iter / copy_tiles_and_advance. Do not change math,
tile, stage count, group scale or defaults. One candidate, not a position sweep.
Prologue remains full-copy; each steady-state prefetch commits exactly once.
"""
from probe_roof_interleaved_merge_codegen import HEADERS, ROOT, main as build

STAGE_COPY = '''    if constexpr(Stages==3) {
      if(stage+2<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages>(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n);
    } else {
      if(stage+1<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels>(s,1-slot,stage+1,a,w,ws,m,k,as,n);
    }
'''
PART_COPY = '''    // Four uniform copy phases. All target a different, barrier-protected
    // stage; only phase 3 commits. The following iteration keeps its old wait.
    auto prefetch_part=[&](auto phase) {
      constexpr int Part=decltype(phase)::value;
      if constexpr(Stages==3) {
        if(stage+2<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages,Part>(s,(stage+2)%Stages,stage+2,a,w,ws,m,k,as,n);
      } else {
        if(stage+1<k/K) o3_prefetch<M,N,K,Fast,Cached,WN,StaticCopy,VectorScale,DualScale,GroupMajorScale,PrebiasActivationScale,AsyncScale,CombinedScalePanels,Stages,Part>(s,1-slot,stage+1,a,w,ws,m,k,as,n);
      }
    };
    prefetch_part(cute::Int<0>{});
'''
COPY_BRANCH = '''  if constexpr(CopyPhase>=0) {
    static_assert(M==64 && N==128 && K==128 && C::Threads==128);
    static_assert(CopyPhase<4 && StaticCopy && !Cached && C::Groups==1);
    constexpr unsigned Step=C::Threads*16;
    unsigned off=threadIdx.x*16;
    if constexpr(CopyPhase==0) copy_a(off);
    if constexpr(CopyPhase==1) {copy_b(off);copy_b(off+Step);}
    if constexpr(CopyPhase==2) copy_a(off+Step);
    if constexpr(CopyPhase==3) {copy_b(off+2*Step);copy_b(off+3*Step);}
  } else if constexpr(StaticCopy) {'''
FIRST_K64_END = '''              });
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);
                cute::gemm(LA{},pl,ra1'''
INSERT_B_COPIES = '''              });
              // First N64 slice has two independent M atoms. Finish all
              // prefetch copies by the first K64 of its second M atom, leaving
              // the remaining MMA/scale work to cover async latency.
              static_assert(MAtoms==2 && NAtoms==4 && N/SliceN==2);
              if constexpr(decltype(nb)::value==0 && decltype(mi)::value==0)
                prefetch_part(cute::Int<1>{});
              if constexpr(decltype(nb)::value==0 && decltype(mi)::value==1)
                prefetch_part(cute::Int<3>{});
              o1_static_for<0,NAtoms>([&](auto ni) {
                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);
                cute::gemm(LA{},pl,ra1'''
FINISH = '''                auto pl=pls(cute::_,ni),ph=phs(cute::_,ni);finish(ni,pl,ph);
              });'''


def once(text, old, new):
    if text.count(old)!=1:
        raise ValueError(f'source drift at {old[:65]!r}')
    return text.replace(old,new)


def generated_header(source, policy):
    if policy not in (0,1):
        raise ValueError('control and one distributed candidate only')
    if policy==0:
        return source
    # Only the prefetch helper receives CopyPhase. Prologue calls use -1;
    # original loops and scale operations are retained for that full-copy case.
    end=source.index('\ntemplate<int M,int N,int K,bool Fast,bool Cached=false')
    helper,body=source[:end],source[end:]
    helper=once(helper,'int Stages=2>\n__device__ __forceinline__ void o3_prefetch',
        'int Stages=2,int CopyPhase=-1>\n__device__ __forceinline__ void o3_prefetch')
    helper=once(helper,'  if constexpr(StaticCopy) {',COPY_BRANCH)
    helper=once(helper,'  if constexpr(AsyncScale) {',
        '  if constexpr(CopyPhase<0 || CopyPhase==0) {\n  if constexpr(AsyncScale) {')
    helper=once(helper,'  asm volatile("cp.async.commit_group;" ::: "memory");',
        '  } // scale panel is issued once, with phase 0\n'
        '  if constexpr(CopyPhase<0 || CopyPhase==3)\n'
        '    asm volatile("cp.async.commit_group;" ::: "memory");')
    body=once(body,STAGE_COPY,PART_COPY)
    body=once(body,FIRST_K64_END,INSERT_B_COPIES)
    body=once(body,FINISH,FINISH+'\n'
        '              if constexpr(decltype(nb)::value==0 && decltype(mi)::value==0)\n'
        '                prefetch_part(cute::Int<2>{});')
    return helper+body


if __name__=='__main__':
    build(transform=generated_header,stem_prefix='distributed_copy',description=__doc__)

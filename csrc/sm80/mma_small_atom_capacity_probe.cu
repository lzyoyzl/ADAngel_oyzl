// v102 capacity diagnostic only. Same physical work, not a GEMM candidate.
// Public INT4 PTX shapes; no production/default/quantization modification.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int Threads=128,Blocks=2048,Shared=50688,Groups=256;
void check(cudaError_t e) {
  if(e!=cudaSuccess) {std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::exit(2);}
}
template<bool U>
__device__ __forceinline__ void large(uint32_t (&p)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  if constexpr(U)
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+r"(p[0]),"+r"(p[1]),"+r"(p[2]),"+r"(p[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  else
    asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+r"(p[0]),"+r"(p[1]),"+r"(p[2]),"+r"(p[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
template<bool U>
__device__ __forceinline__ void small(uint32_t (&p)[2],uint32_t a,uint32_t b) {
  if constexpr(U)
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32 "
      "{%0,%1},{%2},{%3},{%0,%1};"
      : "+r"(p[0]),"+r"(p[1]):"r"(a),"r"(b));
  else
    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
      "{%0,%1},{%2},{%3},{%0,%1};"
      : "+r"(p[0]),"+r"(p[1]):"r"(a),"r"(b));
}
template<bool Small>
__device__ __forceinline__ void body(const uint32_t* input,uint32_t* output,int groups) {
  constexpr int Chains=Small?16:8,Width=Small?2:4,Steps=Small?4:2;
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const volatile uint32_t* src=input+6*tid;
  uint32_t a[4]={src[0],src[1],src[2],src[3]},b[2]={src[4],src[5]};
  extern __shared__ volatile uint32_t occupancy_pad[];
  if((threadIdx.x&31)==0) occupancy_pad[threadIdx.x/32]=a[0];
  uint32_t p[Chains][Width],checksum=0;
  #pragma unroll 1
  for(int group=0;group<groups;++group) {
    #pragma unroll
    for(int slice=0;slice<2;++slice) {
      #pragma unroll
      for(int i=0;i<Chains;++i) for(int j=0;j<Width;++j) p[i][j]=0;
      #pragma unroll
      for(int step=0;step<Steps;++step) {
        #pragma unroll
        for(int i=0;i<Chains;++i) {
          if constexpr(Small) small<false>(p[i],a[0],b[0]);
          else large<false>(p[i],a,b);
        }
      }
      #pragma unroll
      for(int i=0;i<Chains;++i) for(int j=0;j<Width;++j) p[i][j]*=16u;
      #pragma unroll
      for(int step=0;step<Steps;++step) {
        #pragma unroll
        for(int i=0;i<Chains;++i) {
          if constexpr(Small) small<true>(p[i],a[0],b[0]);
          else large<true>(p[i],a,b);
        }
      }
      // Observe ALL D registers: 64 checksum additions/group in both shapes.
      // Unsigned addition defines modulo2^32, including negative dot products.
      #pragma unroll
      for(int i=0;i<Chains;++i) for(int j=0;j<Width;++j) checksum+=p[i][j];
    }
  }
  output[tid]=checksum;
}
}
extern "C" __global__ __launch_bounds__(128)
void adangel_capacity_large_observed(const uint32_t* x,uint32_t* y,int g) {body<false>(x,y,g);}
extern "C" __global__ __launch_bounds__(128)
void adangel_capacity_small_observed(const uint32_t* x,uint32_t* y,int g) {body<true>(x,y,g);}

int main() {
  cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=8 || prop.minor!=0 || prop.multiProcessorCount!=108) return 3;
  const void* kernels[]={reinterpret_cast<const void*>(adangel_capacity_large_observed),
                         reinterpret_cast<const void*>(adangel_capacity_small_observed)};
  int regs[2],local[2],resident[2];
  for(int mode=0;mode<2;++mode) {
    check(cudaFuncSetAttribute(kernels[mode],cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,kernels[mode]));
    regs[mode]=attr.numRegs;local[mode]=attr.localSizeBytes;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident[mode],kernels[mode],Threads,Shared));
    if(resident[mode]!=3) return 4;
  }
  const int count=Blocks*Threads;
  std::vector<uint32_t> input(count*6),host(count);
  uint32_t *x=nullptr,*y=nullptr;
  check(cudaMalloc(&x,input.size()*sizeof(uint32_t)));check(cudaMalloc(&y,count*sizeof(uint32_t)));
  cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
  auto launch=[&](int mode,int groups) {
    void* args[]={&x,&y,&groups};check(cudaLaunchKernel(kernels[mode],dim3(Blocks),dim3(Threads),args,Shared,stream));
  };
  auto set_input=[&](int a,int b) {
    const uint32_t av=0x11111111u*a,bv=0x11111111u*b;
    for(int t=0;t<count;++t) for(int j=0;j<6;++j) input[t*6+j]=j<4?av:bv;
    check(cudaMemcpyAsync(x,input.data(),input.size()*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
    check(cudaStreamSynchronize(stream));
  };
  auto verify=[&](int a,int b,int groups) {
    check(cudaMemcpyAsync(host.data(),y,count*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    const int sa=a<8?a:a-16,sb=b<8?b:b-16;
    const uint32_t expected=static_cast<uint32_t>(64ll*groups*128*(16*sa+a)*sb);
    for(uint32_t got:host) if(got!=expected) {std::fprintf(stderr,"checksum mismatch\n");std::exit(5);}
  };
  int checks=0;
  const int patterns[][2]={{0,15},{1,1},{15,1},{7,8},{8,7},{15,15}};
  for(const auto& p:patterns) {
    set_input(p[0],p[1]);
    for(int mode=0;mode<2;++mode) for(int groups:{1,32}) {
      launch(mode,groups);verify(p[0],p[1],groups);++checks;
    }
  }
  set_input(1,1);
  for(int w=0;w<50;++w) for(int mode=0;mode<2;++mode) launch(mode,Groups);
  check(cudaStreamSynchronize(stream));
  std::vector<float> times[2];
  for(int repeat=0;repeat<200;++repeat) for(int order=0;order<2;++order) {
    const int mode=(repeat+order)%2;
    check(cudaEventRecord(begin,stream));launch(mode,Groups);check(cudaEventRecord(end,stream));
    check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms);
  }
  for(int mode=0;mode<2;++mode) {
    launch(mode,Groups);verify(1,1,Groups);
    std::printf("{\"mode\":%d,\"shape\":[%d,8,%d],\"blocks\":2048,\"threads\":128,\"groups\":256,"
      "\"logical_partial_registers\":32,\"source_chains\":%d,\"mma_per_warp_group\":%d,\"shared_bytes\":%d,"
      "\"registers\":%d,\"local_bytes\":%d,\"active_ctas_per_sm\":%d,\"warmup\":50,"
      "\"validation_checks\":%d,\"checksum_passed\":true,\"raw_ms\":[",
      mode,mode?8:16,mode?32:64,mode?16:8,mode?256:64,Shared,regs[mode],local[mode],resident[mode],checks);
    for(size_t i=0;i<times[mode].size();++i) std::printf("%s%.9g",i?",":"",times[mode][i]);
    std::puts("]}");
  }
  check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
  check(cudaFree(x));check(cudaFree(y));return 0;
}

// Diagnostic only: native MMA issue capacity at three resident CTAs/SM.
// No production binding, real GEMM, source quantization, or experiment MSE.
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr int Threads=128, Blocks=2048, Shared=50688, Groups=256;
void check(cudaError_t e) {
  if(e!=cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(2); }
}
template<bool Unsigned>
__device__ __forceinline__ void mma(uint32_t (&p)[4],const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  if constexpr(Unsigned)
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
template<int Mode>
__device__ __forceinline__ void body(const uint32_t* input,uint32_t* output,int groups) {
  const int tid=blockIdx.x*blockDim.x+threadIdx.x;
  const volatile uint32_t* ptr=input+6*tid;
  uint32_t a[4]={ptr[0],ptr[1],ptr[2],ptr[3]},b[2]={ptr[4],ptr[5]};
  // Actual dynamic shared allocation controls occupancy, not live operands.
  // Each warp writes its own slot once; the MMA loop has no shared access.
  extern __shared__ volatile uint32_t occupancy_pad[];
  if((threadIdx.x&31)==0) occupancy_pad[threadIdx.x/32]=a[0];
  uint32_t p[8][4],checksum=0;
  #pragma unroll 1
  for(int group=0;group<groups;++group) {
    // Two N64-like slices, each eight chains and four MMA/chain.
    #pragma unroll
    for(int slice=0;slice<2;++slice) {
      #pragma unroll
      for(int i=0;i<8;++i) for(int j=0;j<4;++j) p[i][j]=0;
      if constexpr(Mode<2) {
        #pragma unroll
        for(int step=0;step<4;++step) {
          #pragma unroll
          for(int i=0;i<8;++i) mma<Mode==1>(p[i],a,b);
        }
      } else {
        #pragma unroll
        for(int step=0;step<2;++step) {
          #pragma unroll
          for(int i=0;i<8;++i) mma<false>(p[i],a,b);
        }
        #pragma unroll
        for(int i=0;i<8;++i) for(int j=0;j<4;++j) p[i][j]*=16u;
        #pragma unroll
        for(int step=0;step<2;++step) {
          #pragma unroll
          for(int i=0;i<8;++i) mma<true>(p[i],a,b);
        }
      }
      // Observe every chain in every slice/group; dead MMA cannot be erased.
      // Identical16 checksum additions/group in all modes are diagnostic cost.
      #pragma unroll
      for(int i=0;i<8;++i) checksum+=p[i][0];
    }
  }
  output[tid]=checksum;
}
}

extern "C" __global__ __launch_bounds__(128)
void adangel_capacity_signed(const uint32_t* a,uint32_t* y,int g) {body<0>(a,y,g);}
extern "C" __global__ __launch_bounds__(128)
void adangel_capacity_unsigned(const uint32_t* a,uint32_t* y,int g) {body<1>(a,y,g);}
extern "C" __global__ __launch_bounds__(128)
void adangel_capacity_merged(const uint32_t* a,uint32_t* y,int g) {body<2>(a,y,g);}

int main(int argc,char** argv) {
  // Optional exact mode for NCU: fifty target warmups, one timed launch.
  const int profile=argc==2 ? std::atoi(argv[1]) : -1;
  if(argc>2 || profile>2 || profile< -1) return 2;
  cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
  if(prop.major!=8 || prop.minor!=0 || prop.multiProcessorCount!=108) return 3;
  const void* kernels[]={reinterpret_cast<const void*>(adangel_capacity_signed),
    reinterpret_cast<const void*>(adangel_capacity_unsigned),reinterpret_cast<const void*>(adangel_capacity_merged)};
  const char* symbols[]={"adangel_capacity_signed","adangel_capacity_unsigned","adangel_capacity_merged"};
  int regs[3],local_bytes[3];
  for(int mode=0;mode<3;++mode) {
    check(cudaFuncSetAttribute(kernels[mode],cudaFuncAttributeMaxDynamicSharedMemorySize,Shared));
    cudaFuncAttributes attr{};check(cudaFuncGetAttributes(&attr,kernels[mode]));regs[mode]=attr.numRegs;local_bytes[mode]=attr.localSizeBytes;
    int resident=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernels[mode],Threads,Shared));
    // Exact SASS audit separately requires zero hot-loop memory instructions.
    // The observed merged entry spills only an8B output pointer outside it.
    if(resident!=3 || attr.localSizeBytes>8) {std::fprintf(stderr,"unexpected occupancy/local memory\n");return 4;}
  }
  const int count=Blocks*Threads;
  std::vector<uint32_t> input(count*6),host(count);
  uint32_t *x=nullptr,*y=nullptr;check(cudaMalloc(&x,input.size()*sizeof(uint32_t)));check(cudaMalloc(&y,count*sizeof(uint32_t)));
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
  auto verify=[&](int mode,int a,int b,int groups) {
    check(cudaMemcpyAsync(host.data(),y,count*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    const int sa=a<8?a:a-16,sb=b<8?b:b-16;
    const int value=mode==2 ? 128*(16*sa+a)*sb : 256*(mode==1?a:sa)*sb;
    for(uint32_t got:host) if(got!=static_cast<uint32_t>(16*groups*value)) {std::fprintf(stderr,"checksum mismatch\n");std::exit(5);}
  };
  int validation_checks=0;
  if(profile<0) {
    const int patterns[4][2]={{1,1},{15,1},{7,8},{8,7}};
    for(const auto& p:patterns) {
      set_input(p[0],p[1]);
      for(int mode=0;mode<3;++mode) for(int groups:{1,32}) {
        launch(mode,groups);verify(mode,p[0],p[1],groups);++validation_checks;
      }
    }
  }
  set_input(1,1);
  for(int w=0;w<50;++w) for(int mode=0;mode<3;++mode) if(profile<0||mode==profile) launch(mode,Groups);
  check(cudaStreamSynchronize(stream));
  const int repeats=profile<0?200:1;
  std::vector<float> times[3];
  for(int repeat=0;repeat<repeats;++repeat) for(int order=0;order<3;++order) {
    const int mode=(repeat+order)%3;if(profile>=0 && mode!=profile) continue;
    check(cudaEventRecord(begin,stream));launch(mode,Groups);check(cudaEventRecord(end,stream));
    check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms);
  }
  // Checking each mode again is outside all measurements and follows NCU capture.
  for(int mode=0;mode<3;++mode) if(profile<0||mode==profile) {
    launch(mode,Groups);verify(mode,1,1,Groups);
    std::printf("{\"mode\":%d,\"symbol\":\"%s\",\"blocks\":%d,\"threads\":%d,\"groups\":%d,"
      "\"mma_per_warp_group\":64,\"shared_bytes\":%d,\"registers\":%d,\"local_bytes\":%d,\"active_ctas_per_sm\":3,"
      "\"warmup\":50,\"validation_checks\":%d,\"checksum_passed\":true,\"raw_ms\":[",
      mode,symbols[mode],Blocks,Threads,Groups,Shared,regs[mode],local_bytes[mode],validation_checks);
    for(size_t i=0;i<times[mode].size();++i) std::printf("%s%.9g",i?",":"",times[mode][i]);
    std::puts("]}");
  }
  check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
  check(cudaFree(x));check(cudaFree(y));return 0;
}

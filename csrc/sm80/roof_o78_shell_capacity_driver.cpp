// A100 fixed-grid capacity diagnostic. Not a real-trace/MSE experiment.
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>

namespace {
constexpr int M=4096,N=4096,Shared=50688,Groups=256,Threads=128;
const char* names[]={"adangel_capacity_register_shell","adangel_capacity_shared_shell",
                     "adangel_capacity_scaled_shared_shell"};
void check(CUresult status) {
  if(status!=CUDA_SUCCESS) {const char* error=nullptr;cuGetErrorString(status,&error);
    std::fprintf(stderr,"CUDA driver: %s\n",error?error:"unknown");std::exit(2);}
}
int expected(int mode,int row,int col,int groups,int seed) {
  int value=0,a=(row+seed)%5-2,b=(col+seed)%7-3;
  for(int g=0;g<groups;++g) {
    int factor=mode==2?(1+((row+(g&31)+seed)&1))*(3+2*((col+((g&31)>>1)+seed)&1)):1;
    value+=128*a*b*factor;
  }
  return value;
}
}
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  check(cuInit(0));CUdevice device;check(cuDeviceGet(&device,0));
  int major,minor,sms;check(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device));
  check(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device));
  check(cuDeviceGetAttribute(&sms,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,device));
  if(major!=8 || minor!=0 || sms!=108)return 3;
  CUcontext context;check(cuCtxCreate(&context,0,device));
  CUmodule module;check(cuModuleLoad(&module,argv[1]));CUfunction f[3];int regs[3],local[3],resident[3];
  for(int mode=0;mode<3;++mode) {
    check(cuModuleGetFunction(&f[mode],module,names[mode]));
    check(cuFuncSetAttribute(f[mode],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,Shared));
    check(cuFuncGetAttribute(&regs[mode],CU_FUNC_ATTRIBUTE_NUM_REGS,f[mode]));
    check(cuFuncGetAttribute(&local[mode],CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,f[mode]));
    check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident[mode],f[mode],Threads,Shared));
    if(resident[mode]!=3 || regs[mode]>168) {std::fprintf(stderr,"residency gate failed\n");return 4;}
  }
  CUdeviceptr output;check(cuMemAlloc(&output,size_t(M)*N*sizeof(float)));
  CUstream stream;check(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
  CUevent begin,end;check(cuEventCreate(&begin,0));check(cuEventCreate(&end,0));
  std::vector<float> host(size_t(M)*N);
  auto launch=[&](int mode,int groups,int seed) {
    void* args[]={&output,&groups,&seed};
    check(cuLaunchKernel(f[mode],N/128,M/64,1,Threads,1,1,Shared,stream,args,nullptr));
  };
  auto verify=[&](int mode,int groups,int seed) {
    check(cuMemcpyDtoHAsync(host.data(),output,host.size()*sizeof(float),stream));
    check(cuStreamSynchronize(stream));
    // 140 row/column/scale residue combinations; validate all16M outputs.
    int table[10][14];for(int m=0;m<10;++m)for(int n=0;n<14;++n)table[m][n]=expected(mode,m,n,groups,seed);
    for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
      float got=host[size_t(m)*N+n],want=float(table[m%10][n%14]);
      if(!std::isfinite(got) || std::memcmp(&got,&want,sizeof(float))) {
        std::fprintf(stderr,"checksum mismatch mode%d g%d seed%d at%d,%d: %g vs %g\n",mode,groups,seed,m,n,got,want);
        std::exit(5);
      }
    }
  };
  int checks=0;
  for(int seed:{0,4})for(int groups:{1,32,256})for(int mode=0;mode<3;++mode) {
    launch(mode,groups,seed);verify(mode,groups,seed);++checks;
  }
  for(int round=0;round<3;++round) {
    for(int w=0;w<50;++w)for(int order=0;order<3;++order)launch((w+order+round)%3,Groups,0);
    check(cuStreamSynchronize(stream));std::vector<float> times[3];
    for(int repeat=0;repeat<200;++repeat)for(int order=0;order<3;++order) {
      int mode=(repeat+order+round)%3;check(cuEventRecord(begin,stream));launch(mode,Groups,0);
      check(cuEventRecord(end,stream));check(cuEventSynchronize(end));float ms;
      check(cuEventElapsedTime(&ms,begin,end));times[mode].push_back(ms);
    }
    for(int mode=0;mode<3;++mode) {
      launch(mode,Groups,0);verify(mode,Groups,0);
      std::printf("{\"mode\":%d,\"round\":%d,\"symbol\":\"%s\",\"grid\":[32,64],\"threads\":128,"
        "\"groups\":256,\"shared_reserved_bytes\":50688,\"registers\":%d,\"local_bytes\":%d,"
        "\"active_ctas_per_sm\":%d,\"warmup\":50,\"validation_checks\":%d,\"checksum_passed\":true,\"raw_ms\":[",
        mode,round,names[mode],regs[mode],local[mode],resident[mode],checks);
      for(size_t i=0;i<times[mode].size();++i)std::printf("%s%.9g",i?",":"",times[mode][i]);
      std::puts("]}");std::fflush(stdout);
    }
  }
  check(cuEventDestroy(begin));check(cuEventDestroy(end));check(cuStreamDestroy(stream));
  check(cuMemFree(output));check(cuModuleUnload(module));check(cuCtxDestroy(context));return 0;
}

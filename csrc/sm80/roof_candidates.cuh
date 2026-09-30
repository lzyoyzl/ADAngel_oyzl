// Internal prepared-input experiments. Production defaults are untouched.
#pragma once

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(256,2) void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  o3_body<64,128,256,Fast,false,false,2,false,true,false,true,true,true,false,
          DualScale,DualScale,Tune>(a,w,as,ws,y,m,n,k);
}

py::dict benchmark_roof_candidate(std::string variant,int tune,at::Tensor a,at::Tensor as,
    at::Tensor w,at::Tensor ws,int warmup,int repeats) {
  const bool dual=variant=="o7" || variant=="o8";
  TORCH_CHECK(dual || variant=="o3","expected o3/o7/o8");
  TORCH_CHECK(tune>=0 && tune<=3 && warmup>=0 && repeats>0,"invalid candidate/repetitions");
  for(const auto& t : {a,as,w,ws})
    TORCH_CHECK(t.is_cuda() && t.device()==a.device(),"CUDA device mismatch");
  TORCH_CHECK(a.dim()==2 && w.dim()==2 && a.is_contiguous() && w.is_contiguous() &&
      a.scalar_type()==at::kByte && w.scalar_type()==at::kByte,"packed contiguous uint8 required");
  int64_t m64=a.size(0)/2,n64=w.size(0),k64=a.size(1)*2;
  TORCH_CHECK(a.size(0)%2==0 && m64>0 && n64>0 && k64>0 && m64%64==0 && n64%128==0 && k64%256==0 &&
      m64*k64<=2147483647LL && n64*k64<=2147483647LL && m64*n64<=2147483647LL &&
      m64/64<=65535 && n64/128<=65535 && w.size(1)==a.size(1),"invalid aligned shape/index range");
  int m=int(m64),n=int(n64),k=int(k64),g=k/128;
  TORCH_CHECK(as.scalar_type()==at::kFloat,"A scale must be FP32");
  TORCH_CHECK(ws.dim()==2 && ws.size(0)==n && ws.size(1)==g,"invalid W scale shape");
  if(dual) {
    TORCH_CHECK(as.dim()==2 && as.size(0)==m && as.size(1)==g && ws.scalar_type()==at::kFloat &&
        as.stride(0)==1 && as.stride(1)==m && ws.stride(0)==1 && ws.stride(1)==n,
        "dual scales must have physical group-major layout");
  } else {
    TORCH_CHECK(as.dim()==1 && as.size(0)==m && as.is_contiguous() &&
        ws.scalar_type()==at::kByte && ws.is_contiguous(),"O3 row/UE8M0 scale layout required");
  }
  c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,a.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"requires SM80");
  TORCH_CHECK(at::isfinite(as).all().item<bool>() && as.ge(0).all().item<bool>(),"invalid A scale");
  bool fast=false;
  if(dual) TORCH_CHECK(at::isfinite(ws).all().item<bool>() && ws.ge(0).all().item<bool>(),"invalid W scale");
  else {
    TORCH_CHECK(ws.ne(255).all().item<bool>(),"UE8M0 code 255 is invalid");
    float amin=as.min().item<float>(),amax=as.max().item<float>();
    uint32_t lo,hi;std::memcpy(&lo,&amin,4);std::memcpy(&hi,&amax,4);
    int emin=(lo>>23)&255,emax=(hi>>23)&255;
    fast=amin>0 && emin>0 && emin+ws.min().item<int>()-127>=1 && emax+ws.max().item<int>()-127<=254;
  }
  const double maxw=dual ? ws.max().item<double>() : std::ldexp(1.0,ws.max().item<int>()-127);
  TORCH_CHECK(as.max().item<double>()*maxw*double(k)*128.0*8.0<=std::numeric_limits<float>::max(),
      "scale product may overflow output");
  auto y=at::empty({m,n},as.options());
  auto stream=c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const size_t smem=dual ? sizeof(O3AmpereConfig<64,128,256,false,2,true>::Storage)
                         : sizeof(O3AmpereConfig<64,128,256,false,2,false>::Storage);
  auto kernel=adangel_sm80_roof_candidate<false,false,0>;
  #define ROOF_SELECT(T) case T: kernel=dual ? adangel_sm80_roof_candidate<true,false,T> : \
      (fast ? adangel_sm80_roof_candidate<false,true,T> : adangel_sm80_roof_candidate<false,false,T>); break
  switch(tune) {ROOF_SELECT(0);ROOF_SELECT(1);ROOF_SELECT(2);ROOF_SELECT(3);}
  #undef ROOF_SELECT
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
  auto launch=[&]() {
    kernel<<<dim3(n/128,m/64),256,smem,stream>>>(a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),
        as.data_ptr<float>(),reinterpret_cast<const uint8_t*>(ws.data_ptr()),y.data_ptr<float>(),m,n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  for(int j=0;j<warmup;++j) launch();
  auto times=batch(launch,repeats,1,stream);
  py::dict meta;
  meta["kernel_symbol"]="adangel_sm80_roof_candidate";
  meta["tune"]=tune;meta["dual_scale"]=dual;meta["exponent_fast_path"]=fast;
  meta["scale_hoist"]=bool(tune&1);meta["interleaved_n_atoms"]=bool(tune&2);
  meta["cta_tile"]=std::vector<int>{64,128,256};meta["group_size"]=128;
  meta["shared_memory_bytes"]=smem;meta["scope"]="candidate_compute_only_not_production";
  py::dict result;result["output"]=y;result["gemm_ms"]=times;result["kernel"]=meta;
  return result;
}

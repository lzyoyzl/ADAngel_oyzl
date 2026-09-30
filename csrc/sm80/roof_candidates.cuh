// Internal prepared-input experiments. Production defaults are untouched.
#pragma once

// Targeted latency-hiding experiments, not a production tile search:
// 8: twice as many warps at the same CTA shape (register cap from bound2).
// 9: half the output N tile, allowing three resident CTAs if resources permit.
// 10: candidate9 with a G128 stage to bound operand liveness further.
template<int Tune> struct RoofShape {
  static constexpr int N=(Tune==9 || Tune==10)?64:128;
  static constexpr int K=Tune==10?128:256;
  static constexpr int WN=Tune==8?4:2;
  static constexpr int Threads=128*WN;
  static constexpr int MinBlocks=(Tune==9 || Tune==10)?3:2;
  static constexpr int CoreTune=Tune>=11?6:(Tune>=8?2:Tune);
};

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(RoofShape<Tune>::Threads,RoofShape<Tune>::MinBlocks)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  using R=RoofShape<Tune>;
  o3_body<64,R::N,R::K,Fast && !DualScale,false,false,R::WN,false,true,false,true,true,true,false,
          DualScale,DualScale,R::CoreTune,DualScale && Fast && Tune>=11,
          DualScale && Fast && Tune==12>(a,w,as,ws,y,m,n,k);
}

bool valid_roof_tune(int tune) {
  return (tune>=-1 && tune<=3) || (tune>=6 && tune<=12);
}

// Only used outside CUDA Event regions. A power-of-two times a normal W is
// exact iff the resulting exponent remains normal. Conservative global range
// checks deliberately fall back for zero, subnormal, non-power2 or extreme data.
bool roof_power2_activation_guard(const at::Tensor& as,const at::Tensor& ws) {
  if(!at::isfinite(as).all().item<bool>() || !at::isfinite(ws).all().item<bool>() ||
      !as.gt(0).all().item<bool>() || !ws.gt(0).all().item<bool>()) return false;
  auto abits=as.view(at::kInt),wbits=ws.view(at::kInt);
  if(at::bitwise_and(abits,0x007fffff).ne(0).any().item<bool>()) return false;
  int amin=(abits.min().item<int>()>>23)&255,amax=(abits.max().item<int>()>>23)&255;
  int wmin=(wbits.min().item<int>()>>23)&255,wmax=(wbits.max().item<int>()>>23)&255;
  return amin>0 && wmin>0 && amin+wmin-127>=1 && amax+wmax-127<=254;
}

struct RoofLaunchConfig {int n,k,threads,min_blocks,core_tune,slice_n;size_t smem;};
template<int Tune> RoofLaunchConfig roof_config_for(bool dual) {
  using R=RoofShape<Tune>;
  const size_t smem=dual ? sizeof(typename O3AmpereConfig<64,R::N,R::K,false,R::WN,true>::Storage)
                        : sizeof(typename O3AmpereConfig<64,R::N,R::K,false,R::WN,false>::Storage);
  return {R::N,R::K,R::Threads,R::MinBlocks,R::CoreTune,R::WN*((R::CoreTune&4)?32:16),smem};
}
RoofLaunchConfig roof_config(int tune,bool dual) {
  TORCH_CHECK(valid_roof_tune(tune),"invalid roof candidate configuration");
  switch(tune) {
    case 8:return roof_config_for<8>(dual);
    case 9:return roof_config_for<9>(dual);
    case 10:return roof_config_for<10>(dual);
    case 11:return roof_config_for<11>(dual);
    case 12:return roof_config_for<12>(dual);
    case 6:return roof_config_for<6>(dual);
    case 7:return roof_config_for<7>(dual);
    case 3:return roof_config_for<3>(dual);
    case 2:return roof_config_for<2>(dual);
    case 1:return roof_config_for<1>(dual);
    default:return roof_config_for<0>(dual);
  }
}

auto select_roof_kernel(bool dual,bool fast,int tune) {
  TORCH_CHECK(tune>=0 && valid_roof_tune(tune),"invalid roof candidate");
  auto kernel=adangel_sm80_roof_candidate<false,false,0>;
  if(tune==12) return dual ? (fast ? adangel_sm80_roof_candidate<true,true,12> :
      adangel_sm80_roof_candidate<true,false,12>) : (fast ? adangel_sm80_roof_candidate<false,true,12> :
      adangel_sm80_roof_candidate<false,false,12>);
  if(tune==11) return dual ? (fast ? adangel_sm80_roof_candidate<true,true,11> :
      adangel_sm80_roof_candidate<true,false,11>) : (fast ? adangel_sm80_roof_candidate<false,true,11> :
      adangel_sm80_roof_candidate<false,false,11>);
  #define ROOF_PICK(T) case T: kernel=dual ? adangel_sm80_roof_candidate<true,false,T> : \
      (fast ? adangel_sm80_roof_candidate<false,true,T> : adangel_sm80_roof_candidate<false,false,T>); break
  switch(tune) {ROOF_PICK(0);ROOF_PICK(1);ROOF_PICK(2);ROOF_PICK(3);ROOF_PICK(6);ROOF_PICK(7);
               ROOF_PICK(8);ROOF_PICK(9);ROOF_PICK(10);}
  #undef ROOF_PICK
  return kernel;
}

py::dict benchmark_roof_candidate(std::string variant,int tune,at::Tensor a,at::Tensor as,
    at::Tensor w,at::Tensor ws,int warmup,int repeats) {
  const bool dual=variant=="o7" || variant=="o8";
  TORCH_CHECK(dual || variant=="o3","expected o3/o7/o8");
  TORCH_CHECK(valid_roof_tune(tune) && warmup>=0 && repeats>0,"invalid candidate/repetitions");
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
  if(dual) {
    TORCH_CHECK(at::isfinite(ws).all().item<bool>() && ws.ge(0).all().item<bool>(),"invalid W scale");
    if(tune>=11) fast=roof_power2_activation_guard(as,ws);
  } else {
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
  const auto cfg=roof_config(tune,dual);
  const size_t smem=cfg.smem;
  auto kernel=adangel_sm80_roof_candidate<false,false,0>;
  const bool existing_dual=tune==-1 && dual;
  if(tune==-1 && !dual) {
    if(fast) kernel=adangel_sm80_o3_swizzled_bound2<64,128,256,true,false,false,2,false,true,false,true,true>;
    else kernel=adangel_sm80_o3_swizzled_bound2<64,128,256,false,false,false,2,false,true,false,true,true>;
  }
  if(tune>=0) kernel=select_roof_kernel(dual,fast,tune);
  if(existing_dual)
    check(cudaFuncSetAttribute(adangel_sm80_split_grouped_major<128,256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
  else check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
  cudaFuncAttributes attributes{};int resident_blocks=0;
  if(existing_dual) {
    check(cudaFuncGetAttributes(&attributes,adangel_sm80_split_grouped_major<128,256>));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,
        adangel_sm80_split_grouped_major<128,256>,cfg.threads,smem));
  } else {
    check(cudaFuncGetAttributes(&attributes,kernel));
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,kernel,cfg.threads,smem));
  }
  auto launch=[&]() {
    if(existing_dual)
      adangel_sm80_split_grouped_major<128,256><<<dim3(n/128,m/64),256,smem,stream>>>(
          a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),as.data_ptr<float>(),ws.data_ptr<float>(),
          y.data_ptr<float>(),m,n,k);
    else kernel<<<dim3(n/cfg.n,m/64),cfg.threads,smem,stream>>>(a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),
        as.data_ptr<float>(),reinterpret_cast<const uint8_t*>(ws.data_ptr()),y.data_ptr<float>(),m,n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  for(int j=0;j<warmup;++j) launch();
  auto times=batch(launch,repeats,1,stream);
  py::dict meta;
  meta["kernel_symbol"]=tune>=0 ? "adangel_sm80_roof_candidate" :
      (dual ? "adangel_sm80_split_grouped_major" : "adangel_sm80_o3_swizzled_bound2");
  meta["tune"]=tune;meta["dual_scale"]=dual;meta["exponent_fast_path"]=fast;
  meta["activation_power2_fast_path"]=dual && tune>=11 && fast;
  meta["activation_power2_guard_fallback"]=dual && tune>=11 && !fast;
  meta["activation_scale_prebias"]=dual && tune==12 && fast;
  meta["scale_hoist"]=tune>=0 && bool(cfg.core_tune&1);meta["interleaved_n_atoms"]=tune>=0 && bool(cfg.core_tune&2);
  meta["stream_n_slice"]=cfg.slice_n;meta["threads"]=cfg.threads;meta["launch_bounds_min_blocks"]=cfg.min_blocks;
  meta["cta_tile"]=std::vector<int>{64,cfg.n,cfg.k};meta["group_size"]=128;
  meta["shared_memory_bytes"]=smem;meta["scope"]="candidate_compute_only_not_production";
  meta["registers_per_thread"]=attributes.numRegs;meta["local_bytes_per_thread"]=attributes.localSizeBytes;
  meta["max_resident_blocks_per_sm"]=resident_blocks;
  py::dict result;result["output"]=y;result["gemm_ms"]=times;result["kernel"]=meta;
  return result;
}

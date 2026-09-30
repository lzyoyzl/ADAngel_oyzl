// Internal prepared-input experiments. Production defaults are untouched.
#pragma once

// Targeted latency-hiding experiments, not a production tile search:
// 8: twice as many warps at the same CTA shape (register cap from bound2).
// 9: half the output N tile, allowing three resident CTAs if resources permit.
// 10: candidate9 with a G128 stage to bound operand liveness further.
// 16/17: keep N128, three G128 copy stages and a three-CTA register budget;
// compare streamed N32/N64 fragments. FP32 group order is unchanged.
// 18/19: exactly the 16/17 body/layout/stages with a two-CTA register budget.
// 20/21: two M warps reuse B fragments across M atoms, K256/two-stage copy.
// 22/23: retain 2x2-warp reuse; K128, bound3, two/three copy stages.
// 26/27: exactly24/25's reduction body, relaxed to a one-CTA register budget.
// 28: exactly22's body, tightened to a four-CTA register budget (no FP32 reassociation).
// 29: pair two global G128 spans into three ring slots; independent G128 math.
// 30/31: same22/23 pipeline; overlap earlier atom finish with trailing MMA.
// 32/33: same22/23 body with compile-time4096 dimensions (host guarded).
// 34/35: independently rounded products, then window4/window32 balanced sum.
template<int Tune> struct RoofShape {
  static constexpr int N=(Tune==9 || Tune==10)?64:128;
  static constexpr int K=(Tune==10 || (Tune>=16 && Tune<=19) || (Tune>=22 && Tune<=23) || (Tune>=28 && Tune<=42))?128:256;
  static constexpr int Stages=((Tune>=16 && Tune<=19) || Tune==23 || Tune==29 || Tune==31 || Tune==33 || Tune==39 || Tune==40 || Tune==42)?3:2;
  static constexpr int WN=Tune==8?4:2;
  static constexpr int Threads=((Tune>=20 && Tune<=23) || (Tune>=28 && Tune<=42))?128:128*WN;
  static constexpr int MinBlocks=Tune==28?4:((Tune==26 || Tune==27 || Tune==34 || Tune==35 || Tune==36 || Tune==37 || Tune==38)?1:((Tune==9 || Tune==10 || Tune==16 || Tune==17 || (Tune>=22 && Tune<=23) || (Tune>=29 && Tune<=33) || (Tune>=39 && Tune<=42))?3:2));
  static constexpr int CoreTune=(Tune==16 || Tune==18 || Tune==20 || Tune==40)?2:(Tune>=11?6:(Tune>=8?2:Tune));
};

template<bool DualScale,bool Fast,int Tune>
__global__ __launch_bounds__(RoofShape<Tune>::Threads,RoofShape<Tune>::MinBlocks)
void adangel_sm80_roof_candidate(
    const uint8_t* a,const uint8_t* w,const float* as,const uint8_t* ws,
    float* y,int m,int n,int k) {
  using R=RoofShape<Tune>;
  static_assert(R::Stages==2,"three-stage kernels must compile in their own translation unit");
  o3_body<64,R::N,R::K,Fast && !DualScale,false,false,R::WN,false,true,false,true,true,true,false,
          DualScale,DualScale || Tune==13,R::CoreTune,DualScale && Fast && (Tune==11 || Tune==12),
          DualScale && Fast && Tune==12,Tune==14 || Tune==15,Tune==15>(a,w,as,ws,y,m,n,k);
}

bool valid_roof_tune(int tune) {
  return (tune>=-1 && tune<=3) || (tune>=6 && tune<=46);
}

//43/44 are conversion-only variants of41/42, NOT new GEMM instantiations.
bool roof_fused_payload(int tune) {return tune==43 || tune==44;}
bool roof_grouped_payload(int tune) {return tune>=41 && tune<=46;}
void roof_pack_payload(const at::Tensor& src,const at::Tensor& dst,int planes,int rows,int k,cudaStream_t stream) {
  adangel_sm80_experiment::pack_g128_payload(src.data_ptr<uint8_t>(),dst.data_ptr<uint8_t>(),planes,rows,k,stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void roof_payload_metadata(py::dict& meta,int tune,int m,int n,int k) {
  const bool packed=roof_grouped_payload(tune);
  const bool fused=roof_fused_payload(tune);
  meta["payload_layout"]=packed?"plane_group_row_k64_bytes":"plane_row_group_k64_bytes";
  meta["payload_reorder_in_conversion"]=packed;
  meta["payload_reorder_fused"]=fused;
  meta["gemm_tune"]=fused?tune-2:tune;
  meta["conversion_kernels_per_operand"]=packed && !fused?2:1;
  meta["activation_payload_reorder_traffic_bytes"]=packed && !fused?int64_t(2)*m*k:0;
  meta["weight_payload_reorder_traffic_bytes"]=packed && !fused?int64_t(n)*k:0;
}

void check_roof_fixed_shape(int tune,int64_t m,int64_t n,int64_t k) {
  TORCH_CHECK((tune!=32 && tune!=33) || (m==4096 && n==4096 && k==4096),
      "fixed4096 candidate32/33 requires M=N=K=4096; no fallback");
}

// Natural U8 [N,G] -> group-major [G,N]. Called inside O3 W conversion for
// cold/conversion-only; cached before compute-only and steady-state. Values,
// E8M0 decoding, Q4 payload and ordered G128 accumulation do not change.
__global__ void adangel_sm80_roof_reorder_o3_scale(const uint8_t* src,uint8_t* dst,int n,int groups) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n*groups) dst[i]=src[(i%n)*groups+i/n];
}

void roof_reorder_o3_scale(const at::Tensor& src,const at::Tensor& dst,int n,int groups,cudaStream_t stream) {
  adangel_sm80_roof_reorder_o3_scale<<<(n*groups+255)/256,256,0,stream>>>(
      src.data_ptr<uint8_t>(),dst.data_ptr<uint8_t>(),n,groups);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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

struct RoofLaunchConfig {int n,k,threads,min_blocks,core_tune,slice_n,stages;size_t smem;};
template<int Tune> RoofLaunchConfig roof_config_for(bool dual) {
  using R=RoofShape<Tune>;
  size_t smem;
  if constexpr(Tune==39 || Tune==40) {
    smem=adangel_sm80_experiment::fragment_tree_shared_bytes(dual);
  } else if constexpr(Tune==29) {
    smem=adangel_sm80_experiment::paired_pipeline_shared_bytes(dual);
  } else if constexpr(R::Stages==3) {
    smem=adangel_sm80_experiment::three_stage_shared_bytes(dual);
  } else {
    smem=dual ? sizeof(typename O3AmpereConfig<64,R::N,R::K,false,R::WN,true>::Storage)
              : sizeof(typename O3AmpereConfig<64,R::N,R::K,false,R::WN,false>::Storage);
  }
  return {R::N,R::K,R::Threads,R::MinBlocks,R::CoreTune,R::WN*((R::CoreTune&4)?32:16),R::Stages,smem};
}
RoofLaunchConfig roof_config(int tune,bool dual) {
  TORCH_CHECK(valid_roof_tune(tune),"invalid roof candidate configuration");
  switch(tune) {
    case 8:return roof_config_for<8>(dual);
    case 9:return roof_config_for<9>(dual);
    case 10:return roof_config_for<10>(dual);
    case 11:return roof_config_for<11>(dual);
    case 12:return roof_config_for<12>(dual);
    case 13:return roof_config_for<13>(dual);
    case 14:return roof_config_for<14>(dual);
    case 15:return roof_config_for<15>(dual);
    case 16:return roof_config_for<16>(dual);
    case 17:return roof_config_for<17>(dual);
    case 18:return roof_config_for<18>(dual);
    case 19:return roof_config_for<19>(dual);
    case 20:return roof_config_for<20>(dual);
    case 21:return roof_config_for<21>(dual);
    case 22:return roof_config_for<22>(dual);
    case 23:return roof_config_for<23>(dual);
    case 24:return roof_config_for<24>(dual);
    case 25:return roof_config_for<25>(dual);
    case 26:return roof_config_for<26>(dual);
    case 27:return roof_config_for<27>(dual);
    case 28:return roof_config_for<28>(dual);
    case 29:return roof_config_for<29>(dual);
    case 30:return roof_config_for<30>(dual);
    case 31:return roof_config_for<31>(dual);
    case 32:return roof_config_for<32>(dual);
    case 33:return roof_config_for<33>(dual);
    case 34:return roof_config_for<34>(dual);
    case 35:return roof_config_for<35>(dual);
    case 36:return roof_config_for<36>(dual);
    case 37:return roof_config_for<37>(dual);
    case 38:return roof_config_for<38>(dual);
    case 39:return roof_config_for<39>(dual);
    case 40:return roof_config_for<40>(dual);
    case 41:return roof_config_for<41>(dual);
    case 42:return roof_config_for<42>(dual);
    case 43:return roof_config_for<41>(dual);
    case 44:return roof_config_for<42>(dual);
    case 45:case 46:return {64,128,128,4,6,64,tune==45?2:3,
        adangel_sm80_experiment::narrow_payload_shared_bytes(dual,tune)};
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
  if(tune==45 || tune==46) return adangel_sm80_experiment::select_narrow_payload_kernel(dual,fast,tune);
  if(roof_grouped_payload(tune)) return adangel_sm80_experiment::select_grouped_payload_kernel(dual,fast,roof_fused_payload(tune)?tune-2:tune);
  if(tune==39 || tune==40) return adangel_sm80_experiment::select_fragment_tree_kernel(dual,fast,tune);
  if(tune==37 || tune==38) return adangel_sm80_experiment::select_static_eager_tree_kernel(dual,fast,tune);
  if(tune==36) return adangel_sm80_experiment::select_eager_tree_kernel(dual,fast,tune);
  if(tune==34 || tune==35)
    return adangel_sm80_experiment::select_product_tree_kernel(dual,fast,tune);
  if(tune==32 || tune==33)
    return adangel_sm80_experiment::select_fixed_shape_kernel(dual,fast,tune);
  if(tune==30 || tune==31)
    return adangel_sm80_experiment::select_finish_pipeline_kernel(dual,fast,tune);
  if(tune==29)
    return adangel_sm80_experiment::select_paired_pipeline_kernel(dual,fast,tune);
  if(tune==28)
    return adangel_sm80_experiment::select_reuse_budget_kernel(dual,fast,tune);
  if(tune>=26)
    return adangel_sm80_experiment::select_reduction_budget_kernel(dual,fast,tune);
  if(tune>=24)
    return adangel_sm80_experiment::select_reduction_kernel(dual,fast,tune);
  if(tune>=22)
    return adangel_sm80_experiment::select_reuse_pipeline_kernel(dual,fast,tune);
  if(tune>=20)
    return adangel_sm80_experiment::select_warp_reuse_kernel(dual,fast,tune);
  if(tune>=16)
    return adangel_sm80_experiment::select_three_stage_kernel(dual,fast,tune);
  auto kernel=adangel_sm80_roof_candidate<false,false,0>;
  if(tune==15) {
    TORCH_CHECK(dual,"combined asynchronous scale panels require O7/O8");
    return adangel_sm80_roof_candidate<true,false,15>;
  }
  if(tune==14) {
    TORCH_CHECK(dual,"asynchronous FP32 scale candidate requires O7/O8");
    return adangel_sm80_roof_candidate<true,false,14>;
  }
  if(tune==13) {
    TORCH_CHECK(!dual,"group-major UE8M0 candidate is O3 only");
    return fast ? adangel_sm80_roof_candidate<false,true,13> : adangel_sm80_roof_candidate<false,false,13>;
  }
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
  TORCH_CHECK(tune!=13 || !dual,"group-major UE8M0 candidate is O3 only");
  TORCH_CHECK((tune!=14 && tune!=15) || dual,"asynchronous FP32 scale candidate requires O7/O8");
  TORCH_CHECK(valid_roof_tune(tune) && warmup>=0 && repeats>0,"invalid candidate/repetitions");
  TORCH_CHECK(!roof_fused_payload(tune),"candidate43/44 require source-format four-mode API; no conversion is performed by prepared-core API");
  for(const auto& t : {a,as,w,ws})
    TORCH_CHECK(t.is_cuda() && t.device()==a.device(),"CUDA device mismatch");
  TORCH_CHECK(a.dim()==2 && w.dim()==2 && a.is_contiguous() && w.is_contiguous() &&
      a.scalar_type()==at::kByte && w.scalar_type()==at::kByte,"packed contiguous uint8 required");
  int64_t m64=a.size(0)/2,n64=w.size(0),k64=a.size(1)*2;
  check_roof_fixed_shape(tune,m64,n64,k64);
  const int required_k=((tune>=16 && tune<=19) || (tune>=22 && tune<=23) || (tune>=28 && tune<=46))?128:256;
  TORCH_CHECK(a.size(0)%2==0 && m64>0 && n64>0 && k64>0 && m64%64==0 && n64%128==0 && k64%required_k==0 &&
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
    if(tune==11 || tune==12) fast=roof_power2_activation_guard(as,ws);
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
  auto launch_ws=tune==13 ? at::empty({g,n},ws.options()) : ws;
  if(tune==13) roof_reorder_o3_scale(ws,launch_ws,n,g,stream);
  auto launch_a=roof_grouped_payload(tune)?at::empty({2,g,m,64},a.options()):a;
  auto launch_w=roof_grouped_payload(tune)?at::empty({g,n,64},w.options()):w;
  if(roof_grouped_payload(tune)) {
    // Prepared-input API is compute-only; the four-mode adapters time these copies.
    roof_pack_payload(a,launch_a,2,m,k,stream);
    roof_pack_payload(w,launch_w,1,n,k,stream);
  }
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
    else kernel<<<dim3(n/cfg.n,m/64),cfg.threads,smem,stream>>>(launch_a.data_ptr<uint8_t>(),launch_w.data_ptr<uint8_t>(),
        as.data_ptr<float>(),reinterpret_cast<const uint8_t*>(launch_ws.data_ptr()),y.data_ptr<float>(),m,n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  for(int j=0;j<warmup;++j) launch();
  auto times=batch(launch,repeats,1,stream);
  py::dict meta;
  meta["kernel_symbol"]=tune>=0 ? "adangel_sm80_roof_candidate" :
      (dual ? "adangel_sm80_split_grouped_major" : "adangel_sm80_o3_swizzled_bound2");
  meta["tune"]=tune;meta["dual_scale"]=dual;meta["exponent_fast_path"]=fast;
  meta["activation_power2_fast_path"]=dual && (tune==11 || tune==12) && fast;
  meta["activation_power2_guard_fallback"]=dual && (tune==11 || tune==12) && !fast;
  meta["activation_scale_prebias"]=dual && tune==12 && fast;
  meta["scale_copy_async"]=tune==14 || tune==15;
  meta["scale_copy_transaction_bytes"]=(tune==14 || tune==15) ? 16 : 0;
  meta["scale_copy_combined_panels"]=tune==15;
  meta["weight_scale_layout"]=dual || tune==13 ? "group_major" : "row_major";
  meta["weight_scale_reorder_bytes"]=tune==13 ? int64_t(2)*n*g : 0;
  meta["scale_hoist"]=tune>=0 && bool(cfg.core_tune&1);meta["interleaved_n_atoms"]=tune>=0 && bool(cfg.core_tune&2);
  meta["stream_n_slice"]=cfg.slice_n;meta["threads"]=cfg.threads;meta["launch_bounds_min_blocks"]=cfg.min_blocks;
  meta["cta_tile"]=std::vector<int>{64,cfg.n,cfg.k};meta["group_size"]=128;
  meta["pipeline_stages"]=cfg.stages;
  const int warp_m=((tune>=20 && tune<=23) || (tune>=28 && tune<=46))?2:4;
  meta["warp_layout"]=std::vector<int>{warp_m,cfg.threads/(32*warp_m)};
  meta["accumulators_per_thread"]=64*cfg.n/cfg.threads;
  meta["fp32_accumulation_chains"]=(tune==24 || tune==26)?2:((tune==25 || tune==27)?4:1);
  meta["fp32_reassociated"]=tune>=24 && tune<=27;
  if(tune>=34 && tune<=40) meta["fp32_reassociated"]=true;
  meta["product_window_groups"]=(tune>=38 && tune<=40)?2:((tune==34 || tune==36 || tune==37)?4:(tune==35?32:0));
  meta["separate_rounded_products"]=tune>=34 && tune<=40;
  meta["eager_product_reduction"]=tune>=36 && tune<=40;
  meta["compile_time_reduction_phase"]=tune>=37 && tune<=40;
  meta["fragment_local_product_tree"]=tune==39 || tune==40;
  meta["leaf_values_per_thread"]=tune==39?16:(tune==40?8:0);
  meta["paired_group_pipeline"]=tune==39 || tune==40;
  meta["paired_g128_copy"]=tune==29;
  meta["interleaved_mma_finish"]=tune==30 || tune==31;
  meta["compile_time_shape"]=(tune==32 || tune==33) ? std::vector<int>{4096,4096,4096} : std::vector<int>{};
  meta["physical_stage_payload_padding_bytes"]=tune==29?128:0;
  meta["group_accumulation"]=(tune>=24 && tune<=27) ? "interleaved_chains_then_balanced_tree" : "ascending_g128_fma";
  if(tune>=34 && tune<=40) meta["group_accumulation"]="rounded_products_then_window_balanced_tree";
  meta["shared_memory_bytes"]=smem;meta["scope"]="candidate_compute_only_not_production";
  meta["registers_per_thread"]=attributes.numRegs;meta["local_bytes_per_thread"]=attributes.localSizeBytes;
  meta["max_resident_blocks_per_sm"]=resident_blocks;
  roof_payload_metadata(meta,tune,m,n,k);
  py::dict result;result["output"]=y;result["gemm_ms"]=times;result["kernel"]=meta;
  if(roof_grouped_payload(tune)) {
    result["packed_activation_g128_major"]=launch_a;
    result["packed_weight_g128_major"]=launch_w;
  }
  if(tune==13) result["converted_weight_scale"]=launch_ws;
  return result;
}

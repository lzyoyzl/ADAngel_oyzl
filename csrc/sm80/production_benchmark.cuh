// Included after MixedSource and Event helpers. SM80 only.
#pragma once

namespace production {
namespace api=::adangel_sm80_production;
void shape(int m,int n,int k) {
  TORCH_CHECK(m>0 && n>0 && k==4096 && m%64==0 && n%128==0,
      "accepted full-K path requires K4096, M%64=N%128=0");
  TORCH_CHECK(int64_t(m)*k<=INT32_MAX && int64_t(n)*k<=INT32_MAX &&
      int64_t(m)*n<=INT32_MAX && m/64<=65535 && n/128<=65535,"index/grid overflow");
}
void device(const at::Tensor& t) {
  cudaDeviceProp p;check(cudaGetDeviceProperties(&p,t.get_device()));
  TORCH_CHECK(p.major==8 && p.minor==0,"production requires SM80");
}
py::dict metadata(bool mixed) {
  cudaFuncAttributes attr;int blocks;check(api::resources(mixed,&attr,&blocks));
  py::dict r;
  r["implementation"]=mixed?"fullk_v99_best_conversion":"fullk_v89_grouped_cta";
  r["kernel_symbol"]=mixed?"adangel_sm80_o78_fullk_streaming":"adangel_sm80_o3_fullk_grouped";
  r["production_default"]=true;r["library"]="CUTLASS CuTe + CUDA";
  r["architecture"]="sm80";r["requested_implementation"]="production";
  r["cta_tile"]=std::vector<int>{64,128,128};r["threads"]=128;
  r["pipeline_stages"]=mixed?2:3;r["cta_order_group_m"]=mixed?1:8;
  r["registers_per_thread"]=attr.numRegs;r["local_size_bytes"]=attr.localSizeBytes;
  r["shared_memory_bytes"]=mixed?34304:50688;r["active_blocks_per_sm"]=blocks;
  r["partial_storage"]="register";r["group_size"]=128;r["group_count"]=32;
  r["tensor_core"]=true;r["mma_family"]="IMMA";
  r["mma_types"]="U4*S4 and S4*S4";r["output_dtype"]="fp32";
  r["guarded_fullk_int32"]=true;r["fallback"]="per_tile_fp32_G128_scale";
  r["group_accumulation"]="guarded_fullk_integer_then_fp32_epilogue";
  r["fp32_reassociated"]=true;r["scale_layout"]="group_major";
  r["payload_layout"]="plane_group_row_k64_bytes";
  r["integer_guard_in_conversion_timing"]=true;
  return r;
}

template<class W,class A,class G>
py::dict measure(std::string mode,int warmup,int repeats,int inner,cudaStream_t stream,
    W weight,A activation,G gemm) {
  TORCH_CHECK(mode=="conversion_only" || mode=="compute_only" || mode=="cold" || mode=="steady_state","invalid mode");
  TORCH_CHECK(warmup>=0 && repeats>0 && repeats<=100000 && inner>0 && inner<=10000,"invalid repetitions");
  const bool w=mode=="conversion_only" || mode=="cold",a=mode!="compute_only",g=mode!="conversion_only";
  std::vector<Mark> marks(g?repeats:0); // all event allocation precedes timing
  auto direct=[&]() {if(mode=="cold")weight();if(mode=="cold" || mode=="steady_state")activation();gemm();};
  if(g) {
    for(int i=0;i<warmup;++i)direct();
    for(auto& e:marks) {
      check(cudaEventRecord(e.start.e,stream));
      if(mode=="cold")weight();
      if(mode=="cold" || mode=="steady_state")activation();
      check(cudaEventRecord(e.a.e,stream));gemm();check(cudaEventRecord(e.end.e,stream));
    }
    check(cudaEventSynchronize(marks.back().end.e));
  }
  py::dict times,counts;std::vector<float> wt,at,gt,tt;
  if(w) {for(int i=0;i<warmup;++i)weight();wt=batch(weight,repeats,inner,stream);times["weight_conversion"]=wt;counts["weight_conversion"]=inner;}
  if(a) {for(int i=0;i<warmup;++i)activation();at=batch(activation,repeats,inner,stream);times["activation_conversion"]=at;counts["activation_conversion"]=inner;}
  if(g) {
    for(auto& e:marks) {gt.push_back(elapsed(e.a,e.end));tt.push_back(mode=="compute_only"?gt.back():elapsed(e.start,e.end));}
    times["gemm"]=gt;counts["gemm"]=1;
  } else {for(int i=0;i<repeats;++i)tt.push_back(wt[i]+at[i]);gemm();check(cudaStreamSynchronize(stream));}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  times["total"]=tt;counts["total"]=g?1:inner;
  py::dict r;r["timings_ms"]=times;r["stage_timing_inner_repeats"]=counts;
  r["total_timing"]=g?"single_execution_cuda_event":"sum_of_batched_stage_samples";
  r["timing_contract_version"]=2;
  r["timing_strategy"]="conversion_amortized_end_to_end_direct";
  r["measurement_order"]="direct_path_then_isolated_conversions";
  r["weight_cached"]=!w;r["activation_prepared"]=!a;return r;
}

py::dict o3(std::string mode,at::Tensor a,at::Tensor as,at::Tensor w,at::Tensor ws,int warmup,int repeats,int inner) {
  TORCH_CHECK(a.is_cuda() && a.dim()==2 && a.scalar_type()==at::kChar,"A must be CUDA INT8 matrix");
  TORCH_CHECK(w.dim()==2 && w.scalar_type()==at::kByte && ws.scalar_type()==at::kByte && as.scalar_type()==at::kFloat,"invalid O3 dtype/shape");
  const int m=a.size(0),n=w.size(0),k=a.size(1);shape(m,n,k);
  TORCH_CHECK(w.size(1)==k/2 && ws.sizes()==at::IntArrayRef({n,32}) && as.sizes()==at::IntArrayRef({m}),"invalid O3 shape");
  for(const auto& t:{a,as,w,ws}) TORCH_CHECK(t.is_cuda() && t.device()==a.device() && t.is_contiguous() && uintptr_t(t.data_ptr())%16==0,"O3 tensors must be contiguous, aligned, on one CUDA device");
  c10::cuda::CUDAGuard guard(a.device());device(a);
  TORCH_CHECK(at::isfinite(as).all().item<bool>() && ws.min().item<int>()>0 && ws.max().item<int>()<255,"finite A scales and normal UE8M0 1..254 required");
  auto stream=c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  auto opt=w.options();auto pa=at::empty({2,32,m,64},opt),pw=at::empty({32,n,64},opt),s=at::empty({32,n},opt);
  auto meta=at::empty({33,n},opt.dtype(at::kInt)),status=at::empty({n/128},opt.dtype(at::kInt));
  auto out=at::empty({m,n},as.options());auto info=metadata(false);
  auto weight=[&]() {adangel_sm80_experiment::tiled_o3_conversion(false,true,w.data_ptr<uint8_t>(),ws.data_ptr<uint8_t>(),pw.data_ptr<uint8_t>(),s.data_ptr<uint8_t>(),n,k,stream);api::o3_prepare(s.data_ptr<uint8_t>(),meta.data_ptr<int32_t>(),reinterpret_cast<uint32_t*>(status.data_ptr<int32_t>()),n,stream);};
  auto activation=[&]() {adangel_sm80_experiment::tiled_o3_conversion(true,true,reinterpret_cast<const uint8_t*>(a.data_ptr<int8_t>()),nullptr,pa.data_ptr<uint8_t>(),nullptr,m,k,stream);};
  auto gemm=[&]() {api::o3_gemm(pa.data_ptr<uint8_t>(),pw.data_ptr<uint8_t>(),as.data_ptr<float>(),s.data_ptr<uint8_t>(),meta.data_ptr<int32_t>(),reinterpret_cast<const uint32_t*>(status.data_ptr<int32_t>()),out.data_ptr<float>(),m,n,stream);};
  weight();activation();C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(status.max().item<int>()<=1,"O3 source guard rejected input");
  auto r=measure(mode,warmup,repeats,inner,stream,weight,activation,gemm);
  TORCH_CHECK(at::isfinite(out).all().item<bool>(),"nonfinite O3 output");
  info["fallback_tiles"]=status.eq(1).sum().item<int64_t>()*(m/64);
  info["variant"]="o3";
  info["weight_conversion"]="v36_vector_conversion2_and_fullk_guard";info["activation_conversion"]="v36_vector_conversion2";
  r["output"]=out;r["kernel"]=info;r["guard_status"]=status;r["factor_metadata"]=meta;
  r["conversion_scope"]="source_to_fixed_g128_payload_and_fullk_metadata";
  r["packed_activation_g128_major"]=pa;r["packed_weight_g128_major"]=pw;r["converted_weight_scale"]=s;
  // Backward-compatible diagnostic views, outside all measured intervals.
  r["converted_activation"]=pa.permute({0,2,1,3}).contiguous().reshape({2*m,k/2});
  r["converted_weight"]=pw.permute({1,0,2}).contiguous().reshape({n,k/2});
  return r;
}

py::dict mixed(std::string variant,std::string mode,const MixedSource& w,const MixedSource& a,int warmup,int repeats,int inner) {
  const int m=a.rows,n=w.rows;shape(m,n,a.k);
  c10::cuda::CUDAGuard guard(a.payload.device());device(a.payload);
  TORCH_CHECK(uintptr_t(a.payload.data_ptr())%16==0 && uintptr_t(w.payload.data_ptr())%16==0,"vector conversion requires 16-byte aligned sources");
  auto opt=a.payload.options();auto f=opt.dtype(at::kFloat),i=opt.dtype(at::kInt),l=opt.dtype(at::kLong);
  std::vector<at::Tensor> state={at::empty({2,32,m,64},opt),at::empty({32,n,64},opt),
    at::empty({32,m},f),at::empty({32,n},f),at::empty({32,m},i),at::empty({32,n},i),
    at::empty({m},f),at::empty({n},f),at::empty({m},l),at::empty({n},l),
    at::empty({m},i),at::empty({n},i),at::empty({m},i),at::empty({n},i),
    at::empty({m/64,n/128},i),at::empty({m,n},f),at::empty({m,32},i),at::empty({n,32},i)};
  uint64_t v[18],sa[5],sw[5];for(int j=0;j<18;++j)v[j]=reinterpret_cast<uint64_t>(state[j].data_ptr());
  auto source=[](const MixedSource& s,uint64_t* p) {int j=0;for(auto t:{s.payload,s.scale,s.tensor_scale,s.micro8,s.micro4})p[j++]=t.defined()?reinterpret_cast<uint64_t>(t.data_ptr()):0;};
  source(a,sa);source(w,sw);
  // Exact effective source multipliers, read before timing. Row converter still
  // reads the original tensor scale for its FP32 fallback path.
  float am=a.kind==MixedKind::Mx8?4.f:a.tensor_scale.item<float>()*.25f;
  float wm=w.kind==MixedKind::Hif4?1.f:w.tensor_scale.item<float>();
  auto stream=c10::cuda::getCurrentCUDAStream(a.payload.get_device()).stream();
  const int variant_id=variant=="o7"?7:8;auto info=metadata(true);
  auto weight=[&]() {api::mixed_convert(variant_id,false,sw,v,m,n,wm,stream);};
  auto activation=[&]() {api::mixed_convert(variant_id,true,sa,v,m,n,am,stream);};
  auto gemm=[&]() {api::mixed_gemm(v,m,n,stream);};
  weight();activation();C10_CUDA_KERNEL_LAUNCH_CHECK();
  TORCH_CHECK(state[14].max().item<int>()<=1,"mixed source guard rejected invalid input");
  auto r=measure(mode,warmup,repeats,inner,stream,weight,activation,gemm);
  TORCH_CHECK(at::isfinite(state[15]).all().item<bool>(),"nonfinite mixed output");
  info["fallback_tiles"]=state[14].eq(1).sum().item<int64_t>();
  info["variant"]=variant;info["experiment_naming_version"]=3;
  info["paired_fp16_baseline"]=variant_id==7?"o5":"o6";
  info["weight_source_format"]=variant_id==7?"nvfp4_g128":"hif4_g128";
  info["activation_source_format"]=variant_id==7?"mxfp8_e4m3_g128":"nvstyle_fp6_e2m3_g128";
  info["weight_conversion"]=variant_id==7?"v118_packed_NVFP4":"v138_packed_HiF4";
  info["activation_conversion"]=variant_id==7?"v106_MXFP8_warp_lookup":"v123_packed_FP6";
  r["output"]=state[15];r["kernel"]=info;r["guard_status"]=state[14];
  r["conversion_scope"]="source_to_fixed_g128_payload_and_fullk_metadata";
  r["packed_activation_g128_major"]=state[0];r["packed_weight_g128_major"]=state[1];
  r["activation_scale"]=state[2].transpose(0,1);r["weight_scale"]=state[3].transpose(0,1);
  r["converted_activation"]=py::make_tuple(state[0].permute({0,2,1,3}).contiguous().reshape({2*m,2048}),state[2].transpose(0,1));
  r["converted_weight"]=py::make_tuple(state[1].permute({1,0,2}).contiguous().reshape({n,2048}),state[3].transpose(0,1));
  py::dict diagnostic;const char* names[]={"pa","pw","as","ws","af","wf","ab","wb","an","wn","am","wm","ast","wst","status","y","asq","wsq"};
  for(int j=0;j<18;++j)diagnostic[names[j]]=state[j];r["prepared_state"]=diagnostic;
  return r;
}
}

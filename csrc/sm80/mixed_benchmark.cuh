// Internal O5/O6 path for already encoded source tensors. No source quantizer,
// data-loading fallback, or default-production switch is hidden in this entry.
#pragma once

py::dict benchmark_mixed(std::string variant,std::string mode,
    const py::dict& weight_source,const py::dict& activation_source,
    int warmup,int repeats,int inner,std::string tile) {
  TORCH_CHECK(variant=="o5" || variant=="o6","expected o5 or o6");
  TORCH_CHECK(mode=="conversion_only" || mode=="compute_only" ||
      mode=="cold" || mode=="steady_state","invalid mode");
  TORCH_CHECK(warmup>=0 && repeats>0 && inner>1,"invalid repetition count (conversion inner must exceed one)");
  TORCH_CHECK(tile=="64x64x128" || tile=="64x128x256","unsupported tile");
  const MixedSource w(weight_source),a(activation_source);
  TORCH_CHECK((variant=="o5" && w.kind==MixedKind::Nv4 && a.kind==MixedKind::Mx8) ||
      (variant=="o6" && w.kind==MixedKind::Hif4 && a.kind==MixedKind::Nv6),
      "source formats do not match variant");
  TORCH_CHECK(a.k==w.k && a.payload.device()==w.payload.device(),"K/device mismatch");
  c10::cuda::CUDAGuard guard(a.payload.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,a.payload.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"requires A100 SM80");
  const int m=a.rows,n=w.rows,k=a.k,tn=tile=="64x64x128"?64:128,tk=tn==64?128:256;
  TORCH_CHECK(m%64==0 && n%tn==0 && k%tk==0,"shape must be tile aligned");
  TORCH_CHECK(int64_t(m)*n<=2147483647LL && m/64<=65535 && n/tn<=65535,
      "output/grid outside supported range");
  // All allocations, Python source checks, scale guards and setup precede timing.
  MixedConverted cw(w),ca(a);
  auto y=at::empty({m,n},a.payload.options().dtype(at::kFloat));
  const auto stream=c10::cuda::getCurrentCUDAStream(a.payload.get_device()).stream();
  const size_t smem=tn==64 ? sizeof(O3AmpereConfig<64,64,128,false,2,true>::Storage)
                          : sizeof(O3AmpereConfig<64,128,256,false,2,true>::Storage);
  auto kernel=tn==64 ? adangel_sm80_split_grouped<64,128> : adangel_sm80_split_grouped<128,256>;
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
  auto cvw=[&]() {launch_mixed_conversion(w,cw,stream);};
  auto cva=[&]() {launch_mixed_conversion(a,ca,stream);};
  auto gemm=[&]() {
    kernel<<<dim3(n/tn,m/64),256,smem,stream>>>(
        ca.packed.data_ptr<uint8_t>(),cw.packed.data_ptr<uint8_t>(),
        ca.scale.data_ptr<float>(),cw.scale.data_ptr<float>(),y.data_ptr<float>(),m,n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  cvw();cva();
  const double bound=ca.scale.max().item<double>()*cw.scale.max().item<double>()*double(k)*128*8;
  TORCH_CHECK(bound<=double(std::numeric_limits<float>::max()),"scale product may overflow output");
  gemm();  // Defined correctness output even for conversion_only; outside timing.
  const bool weight=mode=="cold" || mode=="conversion_only";
  const bool activation=mode!="compute_only", compute=mode!="conversion_only";
  auto whole=[&]() {if(weight) cvw(); if(activation) cva(); if(compute) gemm();};
  for(int i=0;i<warmup;++i) whole();
  // Dual-track measurement: conversion-only samples are amortized. GPU end-to-end
  // samples below always execute the actual sequence ONCE, never sum medians.
  auto wt=weight ? batch(cvw,repeats,inner,stream) : std::vector<float>{};
  auto at=activation ? batch(cva,repeats,inner,stream) : std::vector<float>{};
  std::vector<float> gt,totals;
  if(!compute) {
    totals=batch(whole,repeats,inner,stream);
  } else if(mode=="compute_only") {
    gt=batch(gemm,repeats,1,stream);totals=gt;
  } else {
    // Event creation and vector allocation are outside all measured intervals.
    std::vector<Mark> marks(repeats);
    for(auto& e:marks) {
      check(cudaEventRecord(e.start.e,stream));
      if(weight) cvw();
      cva();
      check(cudaEventRecord(e.a.e,stream));
      gemm();
      check(cudaEventRecord(e.end.e,stream));
    }
    check(cudaEventSynchronize(marks.back().end.e));
    for(auto& e:marks) {
      gt.push_back(elapsed(e.a,e.end));totals.push_back(elapsed(e.start,e.end));
    }
  }
  py::dict timings,counts;
  if(weight) {timings["weight_conversion"]=wt;counts["weight_conversion"]=inner;}
  if(activation) {timings["activation_conversion"]=at;counts["activation_conversion"]=inner;}
  if(compute) {timings["gemm"]=gt;counts["gemm"]=1;}
  timings["total"]=totals;counts["total"]=compute?1:inner;
  py::dict meta;
  meta["status"]="source_format_path_pending_formal_data_acceptance";
  meta["implementation"]="dual_g128_split_stream_"+tile;
  meta["kernel_symbol"]="adangel_sm80_split_grouped";
  meta["cta_tile"]=std::vector<int>{64,tn,tk};
  meta["mma"]="m16n8k64.u4.s4 + m16n8k64.s4.s4";
  meta["group_size"]=128;meta["partial_storage"]="register";
  meta["data_movement"]="cp.async";meta["pipeline_stages"]=2;
  meta["scale_formula"]="round_fp32(A_fixed_scale[m,g]*W_fixed_scale[n,g])";
  meta["group_accumulation"]="ascending G128, fp32 fma";
  meta["shared_memory_bytes"]=smem;meta["threads"]=256;meta["output_dtype"]="fp32";
  meta["weight_source_format"]=weight_source["format"];
  meta["activation_source_format"]=activation_source["format"];
  py::dict result;
  result["output"]=y;result["timings_ms"]=timings;result["kernel"]=meta;
  result["converted_weight"]=py::make_tuple(cw.packed,cw.scale);
  result["converted_activation"]=py::make_tuple(ca.packed,ca.scale);
  result["stage_timing_inner_repeats"]=counts;
  result["weight_cached"]=!weight;result["activation_prepared"]=!activation;
  result["conversion_scope"]="source_format_to_fixed_only";
  result["total_timing"]=compute?"single_execution_cuda_event":"batched_amortized_cuda_event";
  return result;
}

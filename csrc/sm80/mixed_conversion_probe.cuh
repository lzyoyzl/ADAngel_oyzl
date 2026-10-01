// Standalone conversion-only AB probe; not reachable from formal experiment
// dispatch. No allocation/host validation is included in CUDA Event intervals.
#pragma once
py::dict benchmark_mixed_conversion_probe(const py::dict& source,int implementation,
    int warmup,int repeats,int inner) {
  TORCH_CHECK(implementation>=0 && implementation<=3,"conversion implementation must be 0..3");
  TORCH_CHECK(warmup>=0 && repeats>0 && inner>=2,"invalid conversion repetitions");
  const MixedSource s(source);
  c10::cuda::CUDAGuard guard(s.payload.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,s.payload.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"requires A100 SM80");
  TORCH_CHECK(implementation!=3 || s.k/128<=65535,"tiled conversion grid.y exceeds65535");
  MixedConverted d(s,true);
  auto packed=s.weight?at::empty({s.k/128,s.rows,64},s.payload.options()):
      at::empty({2,s.k/128,s.rows,64},s.payload.options());
  const auto stream=c10::cuda::getCurrentCUDAStream(s.payload.get_device()).stream();
  using Kind=adangel_sm80_experiment::GroupedSourceKind;
  const Kind kind=s.kind==MixedKind::Nv4?Kind::Nv4:(s.kind==MixedKind::Mx8?Kind::Mx8:
      (s.kind==MixedKind::Hif4?Kind::Hif4:Kind::Nv6));
  auto convert=[&]() {
    if(implementation==0) {
      launch_mixed_conversion(s,d,stream);
      roof_pack_payload(d.packed,packed,s.weight?1:2,s.rows,s.k,stream);
    } else if(implementation==1) launch_mixed_fused_payload(s,d,packed,stream);
    else {
      adangel_sm80_experiment::integer_mixed_fixed(kind,implementation==3,
          s.payload.data_ptr<uint8_t>(),s.scale.data_ptr<uint8_t>(),
          s.tensor_scale.defined()?s.tensor_scale.data_ptr<float>():nullptr,
          s.micro8.defined()?s.micro8.data_ptr<uint8_t>():nullptr,
          s.micro4.defined()?s.micro4.data_ptr<uint8_t>():nullptr,
          packed.data_ptr<uint8_t>(),d.scale.data_ptr<float>(),s.rows,s.k,stream);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  struct ConversionPair {Event start,end;};
  std::vector<ConversionPair> events(repeats);
  std::vector<float> values(repeats);
  for(int j=0;j<warmup;++j) convert();
  check(cudaStreamSynchronize(stream));
  for(auto& e:events) {
    check(cudaEventRecord(e.start.e,stream));
    for(int j=0;j<inner;++j) convert();
    check(cudaEventRecord(e.end.e,stream));
  }
  check(cudaEventSynchronize(events.back().end.e));
  for(int j=0;j<repeats;++j) values[j]=elapsed(events[j].start,events[j].end)/inner;
  py::dict result;result["packed"]=packed;result["scale"]=d.scale;result["timings_ms"]=values;
  result["implementation"]=implementation;result["inner_repeats"]=inner;
  result["payload_layout"]="plane_group_row_k64_bytes";
  result["scale_formula_changed"]=false;result["conversion_kernel_count"]=implementation==0?2:1;
  result["scope"]="conversion_only_probe_not_formal_gemm";
  return result;
}

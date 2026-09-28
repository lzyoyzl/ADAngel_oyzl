// Prepared integer core only. O5/O6 format codecs are a separate contract.
// Included after the Event/batch helpers in o1_o3.cu.
#pragma once

template<int N,int K>
__global__ __launch_bounds__(256,2) void adangel_sm80_split_grouped(
    const uint8_t* a,const uint8_t* w,const float* as,const float* ws,
    float* y,int m,int n,int k) {
  // Fast/Cached/Magic=false; WN=2; Stream/VectorStore/DualScale=true.
  o3_body<64,N,K,false,false,false,2,false,true,false,true,true,true,false,true>(
      a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
}

// Same arithmetic and CTA as the row-major baseline; only FP32 scale addressing
// changes. The converter writes this physical layout directly (no hidden repack).
template<int N,int K>
__global__ __launch_bounds__(256,2) void adangel_sm80_split_grouped_major(
    const uint8_t* a,const uint8_t* w,const float* as,const float* ws,
    float* y,int m,int n,int k) {
  o3_body<64,N,K,false,false,false,2,false,true,false,true,true,true,false,true,true>(
      a,w,as,reinterpret_cast<const uint8_t*>(ws),y,m,n,k);
}

py::dict benchmark_split_grouped(at::Tensor a,at::Tensor as,
    at::Tensor w,at::Tensor ws,int warmup,int repeats,std::string tile) {
  TORCH_CHECK(warmup>=0 && repeats>0,"invalid repetitions");
  TORCH_CHECK(tile=="64x64x128" || tile=="64x128x256","unsupported tile");
  for(const auto& t : {a,as,w,ws}) {
    TORCH_CHECK(t.is_cuda() && t.device()==a.device() && t.is_contiguous(),
        "inputs must be contiguous CUDA tensors on the same device");
    TORCH_CHECK(t.dim()==2,"inputs must be rank two");
  }
  TORCH_CHECK(a.scalar_type()==at::kByte && w.scalar_type()==at::kByte,
      "A_split and W_q4 must be packed uint8");
  TORCH_CHECK(as.scalar_type()==at::kFloat && ws.scalar_type()==at::kFloat,
      "decoded A/W group scales must be fp32");
  const int64_t m64=a.size(0)/2,n64=w.size(0),k64=a.size(1)*2;
  TORCH_CHECK(a.size(0)%2==0 && w.size(1)==a.size(1),"invalid packed shape");
  TORCH_CHECK(m64>0 && n64>0 && k64>0 && m64<=65535*64LL &&
      n64<=65535*64LL && k64<=2147483647LL,"shape outside supported range");
  // Device indexing is int32. Reject overflowing address products explicitly.
  TORCH_CHECK(m64*k64<=2147483647LL && n64*k64<=2147483647LL &&
      m64*n64<=2147483647LL,"tensor too large for int32 indexing");
  int m=int(m64),n=int(n64),k=int(k64),tn=tile=="64x64x128"?64:128;
  int tk=tile=="64x64x128"?128:256;
  TORCH_CHECK(m%64==0 && n%tn==0 && k%tk==0,"shape must be tile aligned");
  TORCH_CHECK(as.size(0)==m && ws.size(0)==n &&
      as.size(1)==k/128 && ws.size(1)==k/128,"invalid G128 scale shape");
  c10::cuda::CUDAGuard guard(a.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,a.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"requires SM80 A100");
  for(const auto& t : {as,ws})
    TORCH_CHECK(at::isfinite(t).all().item<bool>() && t.ge(0).all().item<bool>(),
        "group scales must be finite and nonnegative");
  // Conservative finite-output contract, evaluated outside CUDA Event timing.
  double bound=as.max().item<double>()*ws.max().item<double>()*double(k)*128.0*8.0;
  TORCH_CHECK(bound<=double(std::numeric_limits<float>::max()),
      "scale product may overflow fp32 accumulation");
  auto y=at::empty({m,n},as.options());
  auto stream=c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  size_t smem=tile=="64x64x128"
      ? sizeof(O3AmpereConfig<64,64,128,false,2,true>::Storage)
      : sizeof(O3AmpereConfig<64,128,256,false,2,true>::Storage);
  auto kernel=tile=="64x64x128"
      ? adangel_sm80_split_grouped<64,128> : adangel_sm80_split_grouped<128,256>;
  check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
  auto launch=[&]() {
    if(tn==64)
      adangel_sm80_split_grouped<64,128><<<dim3(n/tn,m/64),256,smem,stream>>>(
          a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),as.data_ptr<float>(),
          ws.data_ptr<float>(),y.data_ptr<float>(),m,n,k);
    else
      adangel_sm80_split_grouped<128,256><<<dim3(n/tn,m/64),256,smem,stream>>>(
          a.data_ptr<uint8_t>(),w.data_ptr<uint8_t>(),as.data_ptr<float>(),
          ws.data_ptr<float>(),y.data_ptr<float>(),m,n,k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  for(int i=0;i<warmup;++i) launch();
  auto times=batch(launch,repeats,1,stream);
  py::dict timings;timings["gemm"]=times;timings["total"]=times;
  py::dict meta;
  meta["status"]="prepared_integer_core_only_not_o5_o6_format_validation";
  meta["implementation"]="dual_g128_split_stream_"+tile;
  meta["kernel_symbol"]="adangel_sm80_split_grouped";
  meta["cta_tile"]=std::vector<int>{64,tn,tk};
  meta["scale_shape"]="A[M,K/128], W[N,K/128]";
  meta["scale_dtype"]="fp32";
  meta["scale_formula"]="round_fp32(A_scale[m,g]*W_scale[n,g])";
  meta["group_accumulation"]="ascending G128, fp32 fma";
  meta["mma"]="m16n8k64.u4.s4 + m16n8k64.s4.s4";
  meta["partial_storage"]="register";
  meta["data_movement"]="cp.async";
  meta["pipeline_stages"]=2;
  meta["shared_memory_bytes"]=smem;
  meta["threads"]=256;
  meta["output_dtype"]="fp32";
  py::dict result;result["output"]=y;result["timings_ms"]=timings;result["kernel"]=meta;
  return result;
}

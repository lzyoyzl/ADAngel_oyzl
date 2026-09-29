// Source-format -> fixed-point conversion only; common source quantization is
// outside the timed path. Included in the SM80 TU, never the SM120 extension.
#pragma once

enum class MixedKind { Nv4, Mx8, Hif4, Nv6 };

__host__ __device__ float mixed_e4m3(uint8_t code) {
  const int mag=code&127, exp=mag>>3, mant=mag&7;
  const float value=exp ? ldexpf(float(8+mant),exp-10) : ldexpf(float(mant),-9);
  return code&128 ? -value : value;
}

template<MixedKind Kind,bool GroupMajor=false>
__global__ void adangel_sm80_mixed_to_fixed(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,uint8_t* packed,float* effective,
    int pairs,int rows,int groups_per_row) {
  const int pair=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(pair>=pairs) return;
  const int group=pair/64, in_group=pair%64;
  int q[2];
  if constexpr(Kind==MixedKind::Nv4) {
    const uint8_t byte=payload[pair];
    q[0]=adangel::e2m1_to_q4(byte&15);
    q[1]=adangel::e2m1_to_q4(byte>>4);
  } else if constexpr(Kind==MixedKind::Hif4) {
    const int i8=in_group/4, i4=in_group/2;
    const int e8=(micro8[group*2+i8/8]>>(i8%8))&1;
    const int e4=(micro4[group*4+i4/8]>>(i4%8))&1;
    const uint8_t byte=payload[pair];
    #pragma unroll
    for(int j=0;j<2;++j) {
      const int code=(byte>>(4*j))&15;
      const float mag=ldexpf(float(code&7),e8+e4-2);
      q[j]=__float2int_rn(code&8 ? -mag : mag);
    }
  } else {
    #pragma unroll
    for(int j=0;j<2;++j) {
      const uint8_t code=payload[2*pair+j];
      if constexpr(Kind==MixedKind::Mx8) {
        q[j]=__float2int_rn(__fmul_rn(mixed_e4m3(code),.25f));
      } else {
        // E2M3 -> signed Q6 (F=2); integer q is sign-extended, not zero-padded.
        const int exp=(code&31)>>3, mant=code&7;
        const float mag=exp ? ldexpf(float(8+mant),exp-2) : float(mant)*.5f;
        q[j]=__float2int_rn(code&32 ? -mag : mag);
      }
    }
  }
  if constexpr(Kind==MixedKind::Nv4 || Kind==MixedKind::Hif4) {
    packed[pair]=uint8_t((q[0]&15)|((q[1]&15)<<4));
  } else {
    // One byte per element in total, arranged as [low U4 rows; high S4 rows].
    const uint8_t a=uint8_t(q[0]), b=uint8_t(q[1]);
    packed[pair]=uint8_t((a&15)|((b&15)<<4));
    packed[pairs+pair]=uint8_t((a>>4)|((b>>4)<<4));
  }
  if(in_group==0) {
    const int destination=GroupMajor ? (group%groups_per_row)*rows+group/groups_per_row : group;
    if constexpr(Kind==MixedKind::Hif4) {
      const int code=scale[group];
      effective[destination]=ldexpf(float(4+(code&3)),(code>>2)-50);
    } else if constexpr(Kind==MixedKind::Mx8) {
      // Decode first, multiply second: matches the reference even at exponent -127.
      effective[destination]=__fmul_rn(ldexpf(1.f,int(scale[group])-127),4.f);
    } else {
      float value=__fmul_rn(mixed_e4m3(scale[group]),tensor_scale[0]);
      if constexpr(Kind==MixedKind::Nv6) value=__fmul_rn(value,.25f);
      effective[destination]=value;
    }
  }
}

struct MixedSource {
  MixedKind kind;
  at::Tensor payload,scale,tensor_scale,micro8,micro4;
  int rows,k;
  bool weight;
  explicit MixedSource(const py::dict& source,bool require_fixed=true) {
    // Shared source contract; all scans and scalar reads are OUTSIDE CUDA timing.
    py::module_::import("adangel.quantization.mixed_formats").attr("validate_source")(source);
    const std::string fmt=source["format"].cast<std::string>();
    kind=fmt=="nvfp4_g128" ? MixedKind::Nv4 : fmt=="hif4_g128" ? MixedKind::Hif4
        : fmt=="mxfp8_e4m3_g128" ? MixedKind::Mx8 : MixedKind::Nv6;
    weight=kind==MixedKind::Nv4 || kind==MixedKind::Hif4;
    const auto shape=source["shape"].cast<std::vector<int64_t>>();
    TORCH_CHECK(shape[0]*shape[1]<=2147483647LL,"source exceeds int32 indexing");
    rows=int(shape[0]); k=int(shape[1]);
    payload=source["payload"].cast<at::Tensor>(); scale=source["scale"].cast<at::Tensor>();
    TORCH_CHECK(payload.is_cuda(),"native mixed conversion requires CUDA input");
    if(kind==MixedKind::Hif4) {
      micro8=source["micro8"].cast<at::Tensor>(); micro4=source["micro4"].cast<at::Tensor>();
    } else if(kind==MixedKind::Mx8) {
      TORCH_CHECK(!require_fixed || scale.max().item<int>()<=252,"MXFP8 fixed scale overflow");
    } else {
      tensor_scale=source["tensor_scale"].cast<at::Tensor>();
      const double bound=double(mixed_e4m3(uint8_t(scale.max().item<int>())))
          *tensor_scale.item<double>();
      TORCH_CHECK(bound<=double(std::numeric_limits<float>::max()),"source scale overflow");
    }
  }
};

struct MixedConverted {
  at::Tensor packed,scale;
  bool group_major;
  explicit MixedConverted(const MixedSource& src,bool gm=false):group_major(gm) {
    packed=at::empty({src.weight ? src.rows : 2*src.rows,src.k/2},src.payload.options());
    auto opt=src.payload.options().dtype(at::kFloat);
    // Public logical shape stays [rows,G]; group-major has stride [1,rows].
    scale=gm ? at::empty({src.k/128,src.rows},opt).transpose(0,1)
             : at::empty({src.rows,src.k/128},opt);
  }
};

// Decode the SOURCE format, not its rounded fixed-point approximation. Ordered
// FP32 products then one FP16 RNE cast match dequantize_source(...).half().
template<MixedKind Kind>
__global__ void adangel_sm80_mixed_to_fp16(
    const uint8_t* payload,const uint8_t* scale,const float* tensor_scale,
    const uint8_t* micro8,const uint8_t* micro4,__half* output,int count) {
  const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=count) return;
  const int group=i/128, offset=i%128;
  float local, effective;
  if constexpr(Kind==MixedKind::Nv4) {
    local=adangel::decode_e2m1((payload[i/2]>>(4*(i%2)))&15);
  } else if constexpr(Kind==MixedKind::Hif4) {
    const int code=(payload[i/2]>>(4*(i%2)))&15;
    const int i8=offset/8,i4=offset/4;
    const int e8=(micro8[group*2+i8/8]>>(i8%8))&1;
    const int e4=(micro4[group*4+i4/8]>>(i4%8))&1;
    local=ldexpf(float(code&7),e8+e4-2);
    if(code&8) local=-local;
  } else if constexpr(Kind==MixedKind::Mx8) {
    local=mixed_e4m3(payload[i]);
  } else {
    const int code=payload[i],exp=(code&31)>>3,mant=code&7;
    local=exp?ldexpf(float(8+mant),exp-4):float(mant)*.125f;
    if(code&32) local=-local;
  }
  if constexpr(Kind==MixedKind::Hif4) {
    const int code=scale[group];
    effective=ldexpf(float(4+(code&3)),(code>>2)-50);
  } else if constexpr(Kind==MixedKind::Mx8) {
    effective=ldexpf(1.f,int(scale[group])-127);
  } else {
    effective=__fmul_rn(mixed_e4m3(scale[group]),tensor_scale[0]);
  }
  output[i]=__float2half_rn(__fmul_rn(local,effective));
}

void launch_mixed_fp16(const MixedSource& s,at::Tensor& out,cudaStream_t stream) {
  auto launch=[&](auto tag) {
    adangel_sm80_mixed_to_fp16<decltype(tag)::value><<<((s.rows*s.k)+255)/256,256,0,stream>>>(
        s.payload.data_ptr<uint8_t>(),s.scale.data_ptr<uint8_t>(),
        s.tensor_scale.defined()?s.tensor_scale.data_ptr<float>():nullptr,
        s.micro8.defined()?s.micro8.data_ptr<uint8_t>():nullptr,
        s.micro4.defined()?s.micro4.data_ptr<uint8_t>():nullptr,
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),s.rows*s.k);
  };
  switch(s.kind) {
    case MixedKind::Nv4: launch(std::integral_constant<MixedKind,MixedKind::Nv4>{}); break;
    case MixedKind::Mx8: launch(std::integral_constant<MixedKind,MixedKind::Mx8>{}); break;
    case MixedKind::Hif4: launch(std::integral_constant<MixedKind,MixedKind::Hif4>{}); break;
    case MixedKind::Nv6: launch(std::integral_constant<MixedKind,MixedKind::Nv6>{}); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor dequantize_mixed_source(const py::dict& source) {
  const MixedSource s(source,false);
  c10::cuda::CUDAGuard guard(s.payload.device());
  auto out=at::empty({s.rows,s.k},s.payload.options().dtype(at::kHalf));
  launch_mixed_fp16(s,out,c10::cuda::getCurrentCUDAStream(s.payload.get_device()).stream());
  TORCH_CHECK(at::isfinite(out).all().item<bool>(),"source dequantization overflows FP16");
  return out;
}

void launch_mixed_conversion(const MixedSource& s,MixedConverted& d,cudaStream_t stream) {
  const int pairs=s.rows*s.k/2;
  const dim3 grid((pairs+255)/256);
  auto launch=[&](auto tag) {
    constexpr MixedKind kind=decltype(tag)::value;
    auto layout=[&](auto order) {
    adangel_sm80_mixed_to_fixed<kind,decltype(order)::value><<<grid,256,0,stream>>>(
        s.payload.data_ptr<uint8_t>(),s.scale.data_ptr<uint8_t>(),
        s.tensor_scale.defined()?s.tensor_scale.data_ptr<float>():nullptr,
        s.micro8.defined()?s.micro8.data_ptr<uint8_t>():nullptr,
        s.micro4.defined()?s.micro4.data_ptr<uint8_t>():nullptr,
        d.packed.data_ptr<uint8_t>(),d.scale.data_ptr<float>(),pairs,s.rows,s.k/128);
    };
    if(d.group_major) layout(std::true_type{}); else layout(std::false_type{});
  };
  switch(s.kind) {
    case MixedKind::Nv4: launch(std::integral_constant<MixedKind,MixedKind::Nv4>{}); break;
    case MixedKind::Mx8: launch(std::integral_constant<MixedKind,MixedKind::Mx8>{}); break;
    case MixedKind::Hif4: launch(std::integral_constant<MixedKind,MixedKind::Hif4>{}); break;
    case MixedKind::Nv6: launch(std::integral_constant<MixedKind,MixedKind::Nv6>{}); break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

py::dict convert_mixed_source(const py::dict& source,std::string scale_layout) {
  TORCH_CHECK(scale_layout=="row_major" || scale_layout=="group_major","invalid scale layout");
  const MixedSource s(source);
  c10::cuda::CUDAGuard guard(s.payload.device());
  cudaDeviceProp prop;check(cudaGetDeviceProperties(&prop,s.payload.get_device()));
  TORCH_CHECK(prop.major==8 && prop.minor==0,"mixed conversion currently validated for SM80 only");
  MixedConverted d(s,scale_layout=="group_major");
  launch_mixed_conversion(s,d,c10::cuda::getCurrentCUDAStream(s.payload.get_device()).stream());
  py::dict result; result["packed"]=d.packed; result["scale"]=d.scale;
  result["scale_layout"]=scale_layout;
  result["status"]="conversion_only_not_formal_o5_o6_acceptance";
  return result;
}

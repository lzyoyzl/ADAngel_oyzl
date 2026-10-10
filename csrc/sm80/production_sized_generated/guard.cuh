#pragma once
// Identical norm/product/finite guards; rows are prepared for 4 or 8 groups.
#include <climits>
namespace o78_sized_prepare {
constexpr uint64_t Bound=uint64_t(INT32_MAX)*INT32_MAX;
constexpr uint64_t Cap=Bound+1;
__device__ uint64_t sat_add(uint64_t a,uint64_t b) {return min(a+b,Cap);}
__device__ uint64_t sat_term(uint64_t square,uint64_t factor) {
  const uint64_t f2=factor*factor;
  // Detect multiplication overflow BEFORE admitting its low 64 bits.
  const uint64_t hi=__umul64hi(f2,square),lo=f2*square;
  return hi || lo>Cap?Cap:lo;
}
template<int Kind> __device__ void decode(unsigned c,int& mant,int& exp,bool& bad) {
  bad=false;
  if constexpr(Kind==0) {bad=c==255;mant=1;exp=int(c)-127;}
  else if constexpr(Kind==1) {
    bad=c>126;const int field=c>>3,frac=c&7;
    mant=field?8+frac:frac;exp=field?field-10:-9;
  } else {bad=c==255;mant=4+(c&3);exp=int(c>>2)-50;}
  if(mant) {const int z=__ffs(mant)-1;mant>>=z;exp+=z;}
}
__device__ unsigned warp_max(unsigned x) {
  for(int d=16;d;d>>=1) x=max(x,__shfl_down_sync(0xffffffff,x,d));
  return __shfl_sync(0xffffffff,x,0);
}
__device__ uint64_t warp_sum_sat(uint64_t x) {
  for(int d=16;d;d>>=1) x=sat_add(x,__shfl_down_sync(0xffffffff,x,d));
  return x;
}

} // namespace o78_sized_prepare

extern "C" __global__ __launch_bounds__(128) void adangel_o78_prepare_cta_guard_sized(
    const uint64_t* an,const uint64_t* wn,const int32_t* am,const int32_t* wm,
    const uint32_t* ast,const uint32_t* wst,const float* ab,const float* wb,
    uint32_t* result,int m,int n) {
  __shared__ uint64_t as[4],ws[4];
  __shared__ unsigned af[4],wf[4],st[4],ae[4],we[4];
  const int t=threadIdx.x,lane=t&31,warp=t>>5;
  const int r=blockIdx.y*64+t,c=blockIdx.x*128+t;
  uint64_t a=t<64?an[r]:0,w=wn[c];
  unsigned ax=t<64?unsigned(am[r]):0,wx=unsigned(wm[c]);
  unsigned s=max(t<64?ast[r]:0,wst[c]);
  unsigned axexp=t<64?((__float_as_uint(ab[r])>>23)&255):0;
  unsigned wxexp=(__float_as_uint(wb[c])>>23)&255;
  for(int d=16;d;d>>=1) {
    a=max(a,__shfl_down_sync(0xffffffff,a,d));w=max(w,__shfl_down_sync(0xffffffff,w,d));
    ax=max(ax,__shfl_down_sync(0xffffffff,ax,d));wx=max(wx,__shfl_down_sync(0xffffffff,wx,d));
    s=max(s,__shfl_down_sync(0xffffffff,s,d));
    axexp=max(axexp,__shfl_down_sync(0xffffffff,axexp,d));
    wxexp=max(wxexp,__shfl_down_sync(0xffffffff,wxexp,d));
  }
  if(lane==0) {as[warp]=a;ws[warp]=w;af[warp]=ax;wf[warp]=wx;st[warp]=s;ae[warp]=axexp;we[warp]=wxexp;}
  __syncthreads();
  if(t==0) {
    a=w=0;ax=wx=s=axexp=wxexp=0;
    for(int i=0;i<4;++i) {a=max(a,as[i]);w=max(w,ws[i]);ax=max(ax,af[i]);wx=max(wx,wf[i]);s=max(s,st[i]);axexp=max(axexp,ae[i]);wxexp=max(wxexp,we[i]);}
    // Positive bases have significands <2; exponent sum<=94 ensures that
    // multiplying either signed INT32 extremum by both bases stays finite.
    if(s==0 && (uint64_t(ax)*wx>INT32_MAX || (w && a>o78_sized_prepare::Bound/w) ||
                int(axexp)+int(wxexp)-254>94)) s=1;
    result[blockIdx.y*(n/128)+blockIdx.x]=s;
  }
}

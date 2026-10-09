#include <climits>
namespace o78_prepare {
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

template<int Kind> __device__ void row(
    const uint8_t* packed,const uint8_t* codes,float multiplier,int32_t* factors,
    float* bases,uint64_t* norms,int32_t* maxima,uint32_t* status,int rows,int activation) {
  __shared__ int f[32],verdict,factor_bad;
  __shared__ uint64_t partial[4];
  const int r=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
  if(warp==0) {
    int mant,exp;bool bad;decode<Kind>(codes[r*32+lane],mant,exp,bad);
    int amin=mant?exp:INT_MAX;
    for(int d=16;d;d>>=1) amin=min(amin,__shfl_down_sync(0xffffffff,amin,d));
    amin=__shfl_sync(0xffffffff,amin,0);if(amin==INT_MAX) amin=0;
    const int delta=exp-amin;
    bool overflow=mant && (delta>=31 || uint64_t(mant)>(uint64_t(INT32_MAX)>>delta));
    const int factor=(!bad && !overflow && mant)?int(uint64_t(mant)<<delta):0;
    const unsigned bs=__float_as_uint(multiplier),be=(bs>>23)&255;
    const int newexp=int(be)+amin;
    // float(INT32_MAX) may round to 2^31: activation base <2^97 protects
    // the first epilogue multiplication even for that extremal integer.
    const bool basebad=(bs>>31) || be==0 || be==255 || newexp<=0 || newexp>=255 ||
                       (activation && newexp>223);
    unsigned v=bad?2u:(overflow || basebad?1u:0u);v=warp_max(v);
    if(lane==0) {
      verdict=int(v);status[r]=v;
      bases[r]=v?1.f:__uint_as_float((bs&0x807fffffu)|(unsigned(newexp)<<23));
    }
    f[lane]=factor;
    factors[lane*rows+r]=v?0:factor;
    const unsigned any_overflow=warp_max(unsigned(overflow));
    if(lane==0) factor_bad=int(any_overflow);
    const unsigned fm=warp_max(overflow?unsigned(INT32_MAX):unsigned(factor));
    if(lane==0) maxima[r]=v==2?0:int(fm);
  }
  __syncthreads();
  uint64_t norm=0;
  if(verdict!=2) {
    for(int g=warp;g<32;g+=4) {
      const size_t offset=(size_t(g)*rows+r)*64+lane*2;
      const unsigned low=reinterpret_cast<const uint16_t*>(packed+offset)[0];
      unsigned high=0;
      if(activation) high=reinterpret_cast<const uint16_t*>(packed+size_t(rows)*32*64+offset)[0];
      unsigned sq=0;
      #pragma unroll
      for(int j=0;j<4;++j) {
        int q=int((low>>(4*j))&15);
        if(activation) {q|=((high>>(4*j))&15)<<4;q=(q^128)-128;}
        else q=(q^8)-8;
        sq+=q*q;
      }
      for(int d=16;d;d>>=1) sq+=__shfl_down_sync(0xffffffff,sq,d);
      if(lane==0) norm=sat_add(norm,sat_term(sq,unsigned(f[g])));
    }
  }
  if(lane==0) partial[warp]=norm;
  __syncthreads();
  if(threadIdx.x==0) {
    uint64_t total=0;for(int i=0;i<4;++i) total=sat_add(total,partial[i]);
    // Overflowed factors already select fallback; a capped sentinel is sufficient.
    norms[r]=verdict==2?0:(factor_bad?Cap:total);
  }
}
} // namespace o78_prepare

#define ROW_ENTRY(Name,Kind) \
extern "C" __global__ __launch_bounds__(128) void Name( \
 const uint8_t* p,const uint8_t* c,float mult,int32_t* f,float* b,uint64_t* norm, \
 int32_t* mx,uint32_t* st,int rows,int activation) { \
 o78_prepare::row<Kind>(p,c,mult,f,b,norm,mx,st,rows,activation); }
ROW_ENTRY(adangel_o78_prepare_ue8m0,0)
ROW_ENTRY(adangel_o78_prepare_e4m3,1)
ROW_ENTRY(adangel_o78_prepare_e6m2,2)
#undef ROW_ENTRY

extern "C" __global__ __launch_bounds__(128) void adangel_o78_prepare_cta_guard(
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
    if(s==0 && (uint64_t(ax)*wx>INT32_MAX || (w && a>o78_prepare::Bound/w) ||
                int(axexp)+int(wxexp)-254>94)) s=1;
    result[blockIdx.y*(n/128)+blockIdx.x]=s;
  }
}

#!/usr/bin/env python3
"""v106: compile gate for warp-distributed MXFP8->Q8 conversion only.

Not v34's quotient/remainder decoder, a constant-memory gather or a scale
table. Keep v73 row metadata, timing, weight preparation and the v78 GEMM.
Each lane owns four positive code results; an indexed warp shuffle gathers
the containing word, followed by byte extraction and the original sign.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from analyze_o78_row_fused_codegen import entries
from compare_a100_codegen import compare

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'reports/o378_roof_v73_codegen'


def legacy_q(code):
    """Exact dyadic model of v73 e4m3(code)*.25, followed by RNE.

    Codes 127/255 are invalid E4M3FN source values, rejected by the existing
    source contract. Preserve even the unchecked legacy decoder's numeric
    behavior for these two slots; do not reinterpret them as valid samples.
    """
    if not 0 <= code < 256:
        raise ValueError('byte code required')
    exp, mant = (code & 127) >> 3, code & 7
    magnitude = Fraction(8 + mant, 1) * Fraction(2) ** (exp - 12) if exp else Fraction(mant, 2048)
    return (-1 if code & 128 else 1) * round(magnitude)


def lookup_words():
    return [sum(legacy_q(lane * 4 + byte) << (byte * 8) for byte in range(4)) for lane in range(32)]


def generated_header():
    source = (ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    source = source.replace('namespace row_fused_probe', 'namespace mx8_warp_lut_probe')
    source = source.replace('adangel_sm80_row_conversion_metadata', 'adangel_sm80_row_warp_lut_metadata')
    init = '  unsigned square=0;'
    decode = 'q=__float2int_rn(__fmul_rn(e4m3(c),.25f));'
    if source.count(init) != 1 or source.count(decode) != 1:
        raise ValueError('v73 conversion boundary drift')
    source = source.replace(init, init + '''
  unsigned lut_word=0;
  if constexpr(Format==Kind::Mx8) lut_word=__ldg(mx8_magnitude_words+(threadIdx.x&31));''')
    source = source.replace(decode, 'q=mx8_lookup(c,lut_word);')
    words = ','.join(f'0x{word:08x}u' for word in lookup_words())
    declarations = f'''
// Read-only GLOBAL, not constant memory: the 32 lane loads are coalesced.
// All 32 lanes participate; v73's only early exit is uniform across the CTA.
__device__ const unsigned mx8_magnitude_words[32]={{{words}}};
__device__ __forceinline__ int mx8_lookup(unsigned code,unsigned own_word) {{
  const unsigned word=__shfl_sync(0xffffffffu,own_word,(code&127u)>>2);
  const int value=int((word>>((code&3u)*8u))&255u);
  return code&128u?-value:value;
}}
'''
    marker = 'using namespace vector_probe;'
    if source.count(marker) != 1:
        raise ValueError('namespace boundary drift')
    return source.replace(marker, marker + declarations)


def generated_host():
    old = (ROOT/'csrc/sm80/roof_o78_row_fused_prepare.cu').read_text()
    boundary = 'extern "C" int roof_o78_row_fused_benchmark('
    if old.count(boundary) != 1:
        raise ValueError('old timing boundary drift')
    # A mechanical copy of the EXACT old CUDA Event timing body. Candidate
    # selects activation conversion; all weight paths remain original v73.
    benchmark = old[old.index(boundary):].replace('roof_o78_row_fused_benchmark', 'roof_o78_warp_lut_benchmark')
    benchmark = benchmark.replace('RowFusedOnline x(', 'WarpLutOnline x(')
    prefix = '''// Generated v106; old host/library entries are retained verbatim.
#include "roof_o78_row_fused_prepare.cu"
#include "mx8_warp_lut_generated.cuh"
namespace {
struct WarpLutOnline:RowFusedOnline {
  using RowFusedOnline::RowFusedOnline;
  void weight(bool) { RowFusedOnline::weight(true); }
  void activation(bool candidate) {
    if(!candidate || variant!=7) { RowFusedOnline::activation(true);return; }
    mx8_warp_lut_probe::adangel_sm80_row_warp_lut_metadata<Kind::Mx8,16>
      <<<m,256,0,stream>>>(
        reinterpret_cast<const uint8_t*>(sa[0]),reinterpret_cast<const uint8_t*>(sa[1]),
        reinterpret_cast<const float*>(sa[2]),reinterpret_cast<const uint8_t*>(sa[3]),
        reinterpret_cast<const uint8_t*>(sa[4]),reinterpret_cast<uint8_t*>(v[0]),
        reinterpret_cast<float*>(v[2]),reinterpret_cast<uint32_t*>(v[16]),m,32,a_mult,
        reinterpret_cast<int32_t*>(v[4]),reinterpret_cast<float*>(v[6]),
        reinterpret_cast<uint64_t*>(v[8]),reinterpret_cast<int32_t*>(v[10]),
        reinterpret_cast<uint32_t*>(v[12]));
    adangel_o78_prepare_cta_guard<<<dim3(n/128,m/64),128,0,stream>>>(
      reinterpret_cast<const uint64_t*>(v[8]),reinterpret_cast<const uint64_t*>(v[9]),
      reinterpret_cast<const int32_t*>(v[10]),reinterpret_cast<const int32_t*>(v[11]),
      reinterpret_cast<const uint32_t*>(v[12]),reinterpret_cast<const uint32_t*>(v[13]),
      reinterpret_cast<const float*>(v[6]),reinterpret_cast<const float*>(v[7]),
      reinterpret_cast<uint32_t*>(v[14]),m,n);
  }
};
__global__ void adangel_mx8_warp_lut_exhaustive(int2* out) {
  const unsigned word=__ldg(mx8_warp_lut_probe::mx8_magnitude_words+(threadIdx.x&31));
  for(unsigned shift=0;shift<256;++shift) {
    const unsigned code=(threadIdx.x+shift)&255;
    out[threadIdx.x*256+shift]=make_int2(
      __float2int_rn(__fmul_rn(vector_probe::e4m3(code),.25f)),
      mx8_warp_lut_probe::mx8_lookup(code,word));
  }
}
}
extern "C" int roof_mx8_warp_lut_exhaustive(int2* out,void* stream) {
  if(!out) return 1;
  adangel_mx8_warp_lut_exhaustive<<<1,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(out);
  return int(cudaGetLastError());
}
'''
    return prefix + benchmark


def audit(directory):
    before = (BASELINE/'prepare.sass').read_text()
    after = (directory/'prepare.sass').read_text()
    old, new = entries(before), entries(after)
    comparison = compare(before, after, '|'.join('^'+re.escape(s)+'$' for s in old))
    find = lambda rows, name: next((s,e) for s,e in rows.items()
        if name in s and 'GroupedSourceKindE1E' in s)
    old_symbol, old_entry = find(old, 'adangel_sm80_row_conversion_metadata')
    symbol, new_entry = find(new, 'adangel_sm80_row_warp_lut_metadata')
    resources = {}
    for s, r, stack, shared, local in re.findall(
            r' Function (\S+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',
            (directory/'resources.txt').read_text()):
        resources[s] = dict(registers=int(r), stack=int(stack), shared=int(shared), local=int(local))
    a, b = resources[old_symbol], resources[symbol]
    ops = new_entry['opcode_counts']
    local_count = sum(count for op,count in ops.items() if op.startswith(('LDL','STL')))
    barriers = sum(count for op,count in ops.items() if op.startswith('BAR.SYNC'))
    improvement = 1 - new_entry['instructions']/old_entry['instructions']
    # Predeclared compile potential gate; not a claimed runtime speedup.
    gate = (comparison['passed'] and b['registers']<=a['registers'] and
        b['stack']==b['local']==local_count==0 and b['shared']==256 and barriers==1 and
        improvement>=.05 and ops.get('SHFL.IDX',0)>=16)
    return dict(scope='static_conversion_gate_not_latency_or_MSE', old_controls=comparison,
        control=dict(symbol=old_symbol,**a,**old_entry),candidate=dict(symbol=symbol,**b,**new_entry),
        static_instruction_reduction_fraction=improvement,
        local_instructions=local_count,cta_barriers=barriers,worth_runtime_validation=gate)


def checked(directory):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory/'build.json').read_text())
    for name,digest in r['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('source drift: '+name)
    for name,digest in r['artifact_sha256'].items():
        if sha(directory/name)!=digest: raise ValueError('artifact drift: '+name)
    if (directory/'mx8_warp_lut_generated.cuh').read_text()!=generated_header():
        raise ValueError('generated lookup body drift')
    if (directory/'mx8_warp_lut_prepare.cu').read_text()!=generated_host():
        raise ValueError('CUDA Event wrapper drift')
    if r['lookup_words']!=lookup_words() or r['production_default_changed']:
        raise ValueError('quantization/default drift')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('fresh repository output required')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    original=json.loads((BASELINE/'build.json').read_text())
    for name,digest in original['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('v73 source drift: '+name)
    if sha(BASELINE/'libo78_gpu_prepare.so')!=original['driver_sha256']:
        raise ValueError('v73 library drift')
    cuda=Path('/usr/local/cuda-12.8/bin')
    version=subprocess.check_output([str(cuda/'nvcc'),'--version'],text=True)
    if 'release 12.8' not in version:p.error('CUDA12.8 required')
    out.mkdir(parents=True)
    names=set(original['sources'])|{'scripts/probe_mx8_warp_lut_codegen.py',
        'scripts/analyze_o78_row_fused_codegen.py','scripts/compare_a100_codegen.py'}
    receipt=dict(scope='v106_MXFP8_warp_distributed_exact_lookup_conversion',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:sha(ROOT/s) for s in sorted(names)},lookup_words=lookup_words(),
        nvcc=version,commands=[],GEMM_modified=False,production_default_changed=False,
        invalid_source_policy='existing_source_contract_unchanged',
        low_high_packing_and_row_metadata='verbatim_v73')
    (out/'mx8_warp_lut_generated.cuh').write_text(generated_header())
    (out/'mx8_warp_lut_prepare.cu').write_text(generated_host())
    lib=out/'libo78_gpu_prepare.so'
    def run(command,name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            env=dict(os.environ,TMPDIR=str(ROOT/'tmp'))
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    run([str(cuda/'nvcc'),'-O3','-std=c++17','-lineinfo','-arch=sm_80','-shared',
         '-Xcompiler=-fPIC','-Xptxas=-v','-I'+str(ROOT/'csrc/sm80'),
         str(out/'mx8_warp_lut_prepare.cu'),'-lcuda','-o',str(lib)],'build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(lib)],'prepare.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(lib)],'resources.txt')
    receipt['audit']=audit(out)
    receipt['driver_sha256']=sha(lib)
    receipt['artifact_sha256']={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    checked(out)
    print(json.dumps(receipt['audit'],indent=2),flush=True)


if __name__=='__main__':main()

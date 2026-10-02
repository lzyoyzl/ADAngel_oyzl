#!/usr/bin/env python3
"""v59: predecoded full-K alignment factors, copied with existing cp.async ring.

Compile/audit gate only. Weight metadata is now int32[33,N]: 32 factor rows
followed by original anchor codes. Host guard must still prove INT32 safety.
No mathematical grouping, MMA, output order, or public backend is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from probe_roof_fullk_integer_codegen import static_entries, PATTERN

ROOT = Path(__file__).resolve().parents[1]
PREFETCH_OLD = '''  // ws is [32 original G128 scale rows, 1 anchor row], group-major.
  const int col=blockIdx.x*128+threadIdx.x;
  const uint32_t code=ws[group*n+col],anchor=ws[32*n+col];
  s.factor[slot][threadIdx.x]=int(1u<<(code-anchor));'''
PREFETCH_NEW = '''  // Guarded int32[33,N]: exact factors, then original anchor codes.
  // One warp copies the whole 512B factor panel. Existing commit/wait/barrier
  // covers this panel and A/B together; there is no extra synchronization.
  if(threadIdx.x<32) {
    const auto* factors=reinterpret_cast<const int*>(ws);
    unsigned first=threadIdx.x*4;
    copy16(s.factor[slot]+first,factors+group*n+blockIdx.x*128+first);
  }'''
ANCHOR_OLD = 'const uint32_t code=ws[32*n+blockIdx.x*128+cute::get<1>(p)];'
ANCHOR_NEW = 'const uint32_t code=reinterpret_cast<const uint32_t*>(ws)[32*n+blockIdx.x*128+cute::get<1>(p)];'


def generated_header(source):
    for old, new in ((PREFETCH_OLD, PREFETCH_NEW), (ANCHOR_OLD, ANCHOR_NEW)):
        if source.count(old) != 1:
            raise ValueError('unexpected full-K source; inspect before generating candidate')
        source = source.replace(old, new)
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository directory required')
    cuda = Path('/usr/local/cuda-12.8/bin')
    cutlass = ROOT / 'third_party/cutlass-src'
    nvcc_version = subprocess.check_output([str(cuda/'nvcc'), '--version'], text=True)
    cutlass_commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in nvcc_version or cutlass_commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('CUDA12.8 and pinned CUTLASS required')
    out.mkdir(parents=True)
    header = ROOT / 'csrc/sm80/o3_fullk_integer_probe.cuh'
    wrapper = ROOT / 'csrc/sm80/roof_fullk_integer_probe.cu'
    (out/'o3_factor_async_generated.cuh').write_text(generated_header(header.read_text()))
    wrapper_text = wrapper.read_text()
    needle = '#include "o3_fullk_integer_probe.cuh"'
    if wrapper_text.count(needle) != 1:
        raise ValueError('unexpected full-K wrapper')
    (out/'factor_async.cu').write_text(wrapper_text.replace(needle, '#include "o3_factor_async_generated.cuh"'))
    result = dict(scope='compile_only_not_runtime_or_MSE_or_performance',
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  metadata_layout='int32[33,N]: 32 exact factor rows, 1 UE8M0 anchor code row',
                  sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (header, wrapper)},
                  nvcc=nvcc_version,cutlass_commit=cutlass_commit, variants={}, commands=[])
    def run(cmd, file):
        result['commands'].append(cmd)
        with (out/file).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    for policy in (0,1):
        stem=f'factor_async_{policy}'
        cubin=out/f'{stem}.cubin'
        cmd=[str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
             f'-DADANGEL_FULLK_INTEGER={policy}','-I'+str(cutlass/'include'),
             '-I'+str(ROOT/'csrc/sm80'),str(out/'factor_async.cu')]
        run(cmd+['-cubin','-o',str(cubin),'-Xptxas=-v'],f'{stem}_build.log')
        run(cmd+['-ptx','-o',str(out/f'{stem}.ptx')],f'{stem}_ptx_build.log')
        run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],f'{stem}.sass')
        run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],f'{stem}_resources.txt')
        sass=(out/f'{stem}.sass').read_text()
        entries=static_entries(sass)
        if not all(r['native_u4_s4'] and r['native_s4_s4'] and r['all_copies_bypass_l1'] and not r['int8_mma'] for r in entries.values()):
            raise ValueError('ISA audit failed')
        ptx=(out/f'{stem}.ptx').read_text()
        for block in re.split(r'(?=\.visible \.entry )',ptx):
            if block.startswith('.visible .entry adangel_roof_fullk_integer_'):
                if not all(marker in block for marker in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32')):
                    raise ValueError('same-entry PTX audit failed')
        result['variants'][str(policy)] = dict(entries=entries,cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest())
    old=ROOT/'docs/evidence/a100_o378_roof_v55b/reports/o378_roof_v55b'
    result['control_comparison']=compare((old/'fullk_integer_0.sass').read_text(),(out/'factor_async_0.sass').read_text(),PATTERN)
    result['o78_sentinel']=compare((old/'fullk_integer_0.sass').read_text(),(out/'factor_async_1.sass').read_text(),r'^adangel_roof_fullk_integer_o78$')
    result['previous_fullk_comparison']=compare((old/'fullk_integer_1.sass').read_text(),(out/'factor_async_1.sass').read_text(),PATTERN)
    (out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['control_comparison']['passed'] or not result['o78_sentinel']['passed']:
        raise SystemExit('control/sentinel changed')
    print('compile and same-entry audit passed; resource gate still required')


if __name__ == '__main__': main()

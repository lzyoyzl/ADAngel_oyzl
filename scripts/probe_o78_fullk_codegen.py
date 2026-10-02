#!/usr/bin/env python3
"""v67 isolated compile/audit gate. No formal extension or default changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {'adangel_roof_o78_fullk_control', 'adangel_roof_o78_fullk_candidate'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    cuda = Path('/usr/local/cuda-12.8'); cutlass = ROOT/'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    digest = lambda x: hashlib.sha256(x.read_bytes()).hexdigest()
    sources = [ROOT/'csrc/sm80'/name for name in (
        'o78_fullk_integer_probe.cuh', 'roof_o78_fullk_probe.cu',
        'o78_unsigned_payload_candidate.cuh', 'roof_o78_fullk_driver.cpp',
        'roof_producer_warp_driver.cpp')]
    result = dict(scope='isolated_compile_and_static_audit_not_runtime_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        nvcc=version, cutlass_commit=commit,
        sources={str(x.relative_to(ROOT)):digest(x) for x in sources}, commands=[])
    def run(cmd, filename):
        result['commands'].append(cmd)
        with (out/filename).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    base = [str(cuda/'bin/nvcc'), '-O3','-std=c++17','--expt-relaxed-constexpr',
        '-lineinfo','-arch=sm_80','-I'+str(cutlass/'include'),
        '-I'+str(ROOT/'csrc/sm80'), str(ROOT/'csrc/sm80/roof_o78_fullk_probe.cu')]
    cubin = out/'o78_fullk.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'], 'build.log')
    run(base+['-ptx','-o',str(out/'o78_fullk.ptx')], 'ptx_build.log')
    run([str(cuda/'bin/cuobjdump'),'--dump-sass',str(cubin)], 'o78_fullk.sass')
    run([str(cuda/'bin/cuobjdump'),'--dump-resource-usage',str(cubin)], 'resources.txt')
    entries = static_entries((out/'o78_fullk.sass').read_text(),
        r'^adangel_roof_o78_fullk_(?:control|candidate)$', SYMBOLS)
    assert all(x['native_u4_s4'] and x['native_s4_s4'] and not x['int8_mma'] and
               x['all_copies_bypass_l1'] for x in entries.values())
    ptx = (out/'o78_fullk.ptx').read_text()
    for symbol in SYMBOLS:
        block = next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                     if b.startswith('.visible .entry '+symbol+'('))
        assert all(v in block for v in ('cp.async.cg.shared.global',
                                      '.s32.u4.s4.s32','.s32.s4.s4.s32'))
    library = out/'libo78_fullk_driver.so'
    run(['g++','-O3','-std=c++17','-shared','-fPIC','-I'+str(cuda/'include'),
        str(ROOT/'csrc/sm80/roof_o78_fullk_driver.cpp'),'-lcuda','-o',str(library)],'driver_build.log')
    result.update(entries=entries,cubin_sha256=digest(cubin),driver_sha256=digest(library))
    (out/'codegen.json').write_text(json.dumps(result,indent=2)+'\n')
    print((out/'resources.txt').read_text())
    print('same-entry native INT4 + bypass async-copy audit passed; runtime pending')


if __name__ == '__main__': main()

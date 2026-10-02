#!/usr/bin/env python3
"""v72: compile-only gate for coefficient-first O7/O8 integer dependencies.

The v67 full-K body is the control, not the older per-group FP32 algorithm.
No production build, dispatch switch, or timing claim is made here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ('adangel_roof_o78_coefficient_control', 'adangel_roof_o78_coefficient_candidate')
SOURCE_NAMES = (
    'csrc/sm80/o78_coefficient_first_probe.cuh', 'csrc/sm80/roof_o78_coefficient_probe.cu',
    'csrc/sm80/o78_fullk_integer_probe.cuh', 'csrc/sm80/o78_unsigned_payload_candidate.cuh',
    'csrc/sm80/roof_o78_fullk_driver.cpp', 'csrc/sm80/roof_producer_warp_driver.cpp',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = json.loads((args.baseline/'codegen.json').read_text())
    for path, sha in baseline['sources'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('v67 control source drift: '+path)
    if digest(args.baseline/'o78_fullk.cubin') != baseline['cubin_sha256']:
        raise ValueError('v67 control cubin drift')
    cuda = Path('/usr/local/cuda-12.8')
    cutlass = ROOT/'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    receipt = dict(scope='isolated_O7_O8_compile_only_not_runtime_acceptance',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        sources={name:digest(ROOT/name) for name in SOURCE_NAMES},
        nvcc=version,cutlass_commit=commit,baseline_cubin_sha256=baseline['cubin_sha256'],commands=[])

    def run(command, filename):
        receipt['commands'].append(command)
        with (out/filename).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

    flags = [str(cuda/'bin/nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr',
             '-lineinfo','-arch=sm_80','-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80')]
    base = flags+[str(ROOT/'csrc/sm80/roof_o78_coefficient_probe.cu')]
    cubin = out/'o78_coefficient.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(base+['-ptx','-o',str(out/'o78_coefficient.ptx')],'ptx_build.log')
    run([str(cuda/'bin/cuobjdump'),'--dump-sass',str(cubin)],'o78_coefficient.sass')
    run([str(cuda/'bin/cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    entries = static_entries((out/'o78_coefficient.sass').read_text(),r'^adangel_roof_o78_coefficient_(?:control|candidate)$',set(SYMBOLS))
    for symbol, entry in entries.items():
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma'] and entry['all_copies_bypass_l1'],symbol
    ptx = (out/'o78_coefficient.ptx').read_text()
    for symbol in SYMBOLS:
        body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx) if b.startswith('.visible .entry '+symbol+'('))
        assert all(x in body for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    old = baseline['entries']['adangel_roof_o78_fullk_candidate']
    new_control = entries[SYMBOLS[0]]
    # Relocation/entry names may change; instruction totals and opcodes should not.
    receipt.update(entries=entries,cubin_sha256=digest(cubin),fragment_mapping_unchanged_v67=True,
        control_opcode_counts_match_v67=new_control['opcode_counts']==old['opcode_counts'],
        control_instruction_count_match_v67=new_control['instructions']==old['instructions'],
        production_default_changed=False)
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'resources.txt').read_text())
    for symbol in SYMBOLS:
        print(symbol,entries[symbol]['instructions'],entries[symbol]['opcode_counts'])
    print('native INT4/copy audit passed; inspect resource and scheduling cost before GPU launch')


if __name__ == '__main__':
    main()

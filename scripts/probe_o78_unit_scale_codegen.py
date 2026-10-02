#!/usr/bin/env python3
"""v77 legal unit-scale cost isolation, NOT a new experiment-performance candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from probe_roof_fullk_integer_codegen import static_entries
from inspect_o78_register_liveness import analyze

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'adangel_roof_o78_unit_scale_diagnostic'
CONTROL = 'adangel_roof_o78_fullk_candidate'


def unit_header(source):
    old = '''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];'''
    if source.count(old) != 1:
        raise ValueError('original coefficient source drift')
    return ('// Generated v77 diagnostic. Valid only for host-checked UNIT scales.\n' +
        source.replace('o78_fullk_integer_experiment', 'o78_unit_scale_diagnostic').replace(
            old, '            constexpr int coefficient=1; // unit-input diagnostic ONLY'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    digest = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    prior = json.loads((a.baseline / 'codegen.json').read_text())
    for name, sha in prior['sources'].items():
        if digest(ROOT / name) != sha: raise ValueError('baseline source drift: '+name)
    if digest(a.baseline / 'o78_fullk.cubin') != prior['cubin_sha256']:
        raise ValueError('baseline binary drift')
    cuda, cutlass = Path('/usr/local/cuda-12.8'), ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'bin/nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    header = out / 'o78_unit_scale_generated.cuh'
    header.write_text(unit_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()))
    sources = set(prior['sources']) | {'csrc/sm80/roof_o78_unit_scale_probe.cu', 'scripts/probe_o78_unit_scale_codegen.py'}
    receipt = dict(diagnostic_only=True, performance_candidate=False, default_changed=False,
        valid_inputs='all payload scales, factors and bases one; K4096',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={s:digest(ROOT/s) for s in sorted(sources)}, generated_header_sha256=digest(header),
        control_original_cubin_sha256=prior['cubin_sha256'], nvcc=version, cutlass_commit=commit, commands=[])
    def run(cmd, name):
        receipt['commands'].append(cmd)
        with (out/name).open('w') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    base = [str(cuda/'bin/nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),str(ROOT/'csrc/sm80/roof_o78_unit_scale_probe.cu')]
    cubin = out/'o78_unit_scale.cubin'
    run(base+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(base+['-ptx','-o',str(out/'o78_unit_scale.ptx')],'ptx_build.log')
    run([str(cuda/'bin/cuobjdump'),'--dump-sass',str(cubin)],'o78_unit_scale.sass')
    run([str(cuda/'bin/cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'bin/nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    symbols={SYMBOL, CONTROL, 'adangel_roof_o78_fullk_control'}
    entries=static_entries((out/'o78_unit_scale.sass').read_text(),
        r'^adangel_roof_o78_(?:fullk_(?:control|candidate)|unit_scale_diagnostic)$', symbols)
    assert all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values())
    for symbol in symbols:
        ptx=next(b for b in re.split(r'(?=\.visible \.entry )',(out/'o78_unit_scale.ptx').read_text())
                 if b.startswith('.visible .entry '+symbol+'('))
        assert all(x in ptx for x in ('cp.async.cg.shared.global','.s32.u4.s4.s32','.s32.s4.s4.s32'))
    live=(out/'liveness.txt').read_text()
    receipt.update(cubin_sha256=digest(cubin), entries=entries,
        control_opcode_counts_match=entries[CONTROL]['opcode_counts']==prior['entries'][CONTROL]['opcode_counts'],
        control_instructions_match=entries[CONTROL]['instructions']==prior['entries'][CONTROL]['instructions'],
        liveness={s:analyze(live,s) for s in (CONTROL,SYMBOL)})
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print((out/'resources.txt').read_text())
    print('Unit-scale diagnostic compiled; no original-experiment speedup claimed')


if __name__ == '__main__': main()

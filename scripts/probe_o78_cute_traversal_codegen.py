#!/usr/bin/env python3
"""v88: pinned CuTe register-reuse traversal for the existing eight MMA chains.

Only the four eight-atom phase traversals change. Invoke CuTe's rank-3 D /
rank-2 A,B gemm dispatch instead of hand-written M-major single-atom loops.
Do not assume a source reorder survives ptxas or improves actual reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_eight_chain_schedule import instructions, trace
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o78_cute_traversal_candidate'
STEM = 'o78_cute_traversal'
REFERENCE = 'include/cute/algorithm/gemm.hpp'


def generated_header(source):
    out = eight_header(source)
    for atom, operand, weight in (('HA', 'h0', 'b0'), ('HA', 'h1', 'b1'),
                                   ('LA', 'a0', 'b0'), ('LA', 'a1', 'b1')):
        old = '''      o1_static_for<0,2>([&](auto mi) {
        o1_static_for<0,4>([&](auto ni) {
          auto p=partial(cute::_,mi,ni);
          cute::gemm(ATOM{},p,OPERAND(cute::_,mi,cute::_0{}),WEIGHT(cute::_,ni,cute::_0{}),p);
        });
      });
'''.replace('ATOM', atom).replace('OPERAND', operand).replace('WEIGHT', weight)
        new = ('      // Pinned CuTe dispatch owns the M/N register-reuse traversal.\n'
               f'      cute::gemm({atom}{{}},partial,{operand}(cute::_,cute::_,cute::_0{{}}),\n'
               f'          {weight}(cute::_,cute::_,cute::_0{{}}),partial);\n')
        if out.count(old) != 1:
            raise ValueError('v78 phase boundary drift: ' + operand)
        out = out.replace(old, new)
    return out.replace('o78_eight_chain_experiment', 'o78_cute_traversal_experiment')


def loop_summary(sass, symbol, live):
    loop = next(x for x in live['loops'] if x['kind'] == 'integer')
    ops = instructions(sass, symbol, loop)
    mma = [(pc, text) for pc, text in ops if text.startswith('IMMA.')]
    return dict(instructions=len(ops), opcode_counts=loop['opcode_counts'],
        mma_count=len(mma), operand_reuse_markers=sum(text.count('.reuse') for _, text in mma),
        mma=[dict(pc=hex(pc), instruction=text) for pc, text in mma],
        scope='static_encoded_hints_not_measured_register_bank_traffic')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v78_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    prior = json.loads((a.baseline / 'codegen.json').read_text())
    for path, digest in prior['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('v78 source drift: ' + path)
    if sha(a.baseline / 'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda = Path('/usr/local/cuda-12.8/bin'); cutlass = ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda / 'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source = (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out / 'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header = out / (STEM + '_generated.cuh'); header.write_text(generated_header(source))
    sources = set(prior['sources']) | {'csrc/sm80/roof_o78_cute_traversal_probe.cu',
                                      'scripts/probe_o78_cute_traversal_codegen.py'}
    receipt = dict(scope='CuTe_MMA_traversal_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={s: sha(ROOT / s) for s in sorted(sources)}, generated_header_sha256=sha(header),
        baseline_cubin_sha256=prior['cubin_sha256'], nvcc=version, cutlass_commit=commit,
        cutlass_reference=dict(path=REFERENCE, sha256=sha(cutlass / REFERENCE)),
        production_default_changed=False, cta_tile=[64,128,128], threads=128,
        stages=2, shared_bytes=34304, changed_semantics=False, commands=[])
    def run(cmd, name):
        receipt['commands'].append(cmd)
        with (out / name).open('w') as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    flags = [str(cuda / 'nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-arch=sm_80', '-I'+str(cutlass / 'include'), '-I'+str(ROOT / 'csrc/sm80'), '-I'+str(out)]
    src = str(ROOT / 'csrc/sm80/roof_o78_cute_traversal_probe.cu'); cubin = out / (STEM + '.cubin')
    run(flags + [src, '-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(flags + [src, '-ptx', '-o', str(out / (STEM + '.ptx'))], 'ptx_build.log')
    run([str(cuda / 'cuobjdump'), '--dump-sass', str(cubin)], STEM + '.sass')
    run([str(cuda / 'cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda / 'nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    sass = (out / (STEM + '.sass')).read_text(); ptx = (out / (STEM + '.ptx')).read_text()
    entries = static_entries(sass, '^(?:'+CONTROL+'|'+SYMBOL+')$', {CONTROL, SYMBOL})
    live = {s: analyze((out / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)}
    for s in (CONTROL, SYMBOL):
        body = next(b for b in re.split(r'(?=\.visible \.entry )', ptx) if b.startswith('.visible .entry '+s+'('))
        if not all(x in body for x in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy audit failed')
    receipt.update(entries=entries, liveness=live, cubin_sha256=sha(cubin),
        control_comparison=compare((a.baseline/'o78_eight_chain.sass').read_text(), sass, '^'+CONTROL+'$'),
        schedule={s: trace(sass, s, live[s]) for s in (CONTROL, SYMBOL)},
        loop_summary={s: loop_summary(sass, s, live[s]) for s in (CONTROL, SYMBOL)},
        artifact_sha256={s: sha(out/s) for s in ('build.log', STEM+'.sass', STEM+'.ptx', 'resources.txt', 'liveness.txt')})
    (out / 'codegen.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['control_comparison']['passed'] or not all(e['native_u4_s4'] and e['native_s4_s4']
            and not e['int8_mma'] and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('same-entry INT4/control audit failed')
    print((out / 'build.log').read_text())
    for s in (CONTROL, SYMBOL):
        info = receipt['loop_summary'][s]
        print(json.dumps(dict(symbol=s, registers=live[s]['allocated_gpr'],
            **{k: v for k, v in info.items() if k != 'mma'}), indent=2))
    print('Compile audit only: actual runtime improvement has not been measured.')


if __name__ == '__main__': main()

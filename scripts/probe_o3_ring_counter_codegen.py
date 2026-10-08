#!/usr/bin/env python3
"""v140: recurrent three-slot index, not an unrolled three-group mainloop.

Retain v89's guarded native INT4 math, stage lifetime, CTA order and ABI.
This compile-only gate is deliberately separate from GPU acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import checked as checked_grouped, generated_headers
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o3_grouped_cta_candidate'
SYMBOL = 'adangel_roof_o3_ring_counter_candidate'
STEM = 'o3_ring_counter'
LIMITS = dict(max_instruction_ratio=0.97, max_register_increase=0,
              max_hot_local_instructions=0)


def schedule(groups=32):
    """Oracle of the recurrence, independent of the original modulo formula."""
    if not isinstance(groups, int) or groups < 2:
        raise ValueError('at least two groups required')
    slot = 0
    result = []
    for group in range(groups):
        future = (group + 2, 2 if slot == 0 else slot - 1) if group + 2 < groups else None
        result.append((group, slot, future))
        slot = 0 if slot == 2 else slot + 1
    return result


def generated_header():
    source = generated_headers('o3')[0]
    replacements = (
        ('  for(int group=0;group<groups;++group) {\n    const int slot=group%3;',
         '  int slot=0;\n  #pragma unroll 1\n  for(int group=0;group<groups;++group) {'),
        ('prefetch(s,(group+2)%3,group+2,a,w,ws,m,n,k);',
         'prefetch(s,slot==0 ? 2 : slot-1,group+2,a,w,ws,m,n,k);'),
        ('    });\n  }\n  auto final_value=',
         '    });\n    slot=slot==2 ? 0 : slot+1;\n  }\n  auto final_value='),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('v89 ring boundary drift: ' + old)
        source = source.replace(old, new)
    return source.replace('o3_grouped_cta_experiment', 'o3_ring_counter_experiment')


def cost_gate(control, candidate):
    old = next(x for x in control['loops'] if x['kind'] == 'integer')
    new = next(x for x in candidate['loops'] if x['kind'] == 'integer')
    counts = new['opcode_counts']
    checks = dict(
        work_reduction=new['static_instructions'] <= old['static_instructions'] * LIMITS['max_instruction_ratio'],
        register_capacity=candidate['allocated_gpr'] <= control['allocated_gpr'],
        no_hot_local=not any(v for op, v in counts.items() if op.split('.')[0] in ('LDL', 'STL')),
        same_native_math=counts.get('IMMA.16864.S4.S4') == counts.get('IMMA.16864.U4.S4') == 32,
        same_operand_loads=counts.get('LDSM.16.M88.4') == 16,
        same_copy_and_barrier=all(counts.get(op, 0) == old['opcode_counts'].get(op, 0)
            for op in ('LDGSTS.E.BYPASS.128', 'LDGDEPBAR', 'BAR.SYNC.DEFER_BLOCKING')),
    )
    return dict(passed=all(checks.values()), checks=checks,
                instruction_ratio=new['static_instructions'] / old['static_instructions'],
                limits=LIMITS, interpretation='compile_investment_gate_not_measured_speedup')


def checked(directory):
    r = json.loads((directory / 'codegen.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in r['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('ring-counter source drift: ' + name)
    for name, digest in r['artifact_sha256'].items():
        if sha(directory / name) != digest:
            raise ValueError('ring-counter artifact drift: ' + name)
    if (directory / (STEM + '_generated.cuh')).read_text() != generated_header():
        raise ValueError('generated header drift')
    live = {s: analyze((directory / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)}
    if live != r['liveness'] or cost_gate(live[CONTROL], live[SYMBOL]) != r['cost_gate']:
        raise ValueError('liveness/gate replay drift')
    if not r['control_comparison']['passed'] or any(r[key] for key in (
            'changed_semantics', 'production_default_changed', 'conversion_changed')):
        raise ValueError('control/default/semantics drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('same-entry native INT4/copy audit failed')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v89_o3_codegen'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out, baseline = args.output.resolve(), args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    prior = checked_grouped(baseline, 'o3')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    cuda, cutlass = Path('/usr/local/cuda-12.8/bin'), ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda / 'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA12.8/CUTLASS required')
    out.mkdir(parents=True)
    for f in baseline.iterdir():
        if f.suffix in ('.cu', '.cuh'):
            (out / f.name).write_bytes(f.read_bytes())
    (out / (STEM + '_generated.cuh')).write_text(generated_header())
    sources = set(prior['sources']) | {
        'scripts/probe_o3_ring_counter_codegen.py', 'scripts/compare_a100_codegen.py',
        'scripts/inspect_o78_register_liveness.py', 'scripts/probe_roof_fullk_integer_codegen.py',
        'csrc/sm80/roof_o3_ring_counter_probe.cu',
    }
    r = dict(scope='v140_recurrent_ring_compile_gate_not_GPU_acceptance',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={s: sha(ROOT / s) for s in sorted(sources)}, commands=[], nvcc=version,
        cutlass_commit=commit, baseline_cubin_sha256=prior['cubin_sha256'],
        cta_tile=[64, 128, 128], threads=128, stages=3, shared_bytes=50688,
        partial_registers=32, independent_chains=8, group_m=8, groups_per_loop=1,
        changed_semantics=False, production_default_changed=False, conversion_changed=False)

    def run(command, name):
        r['commands'].append(command)
        (out / 'progress.json').write_text(json.dumps(r, indent=2) + '\n')
        with (out / name).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, TMPDIR=str(ROOT / 'tmp')),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    flags = [str(cuda / 'nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-arch=sm_80', '-DADANGEL_FULLK_INTEGER=1', '-I' + str(cutlass / 'include'),
        '-I' + str(ROOT / 'csrc/sm80'), '-I' + str(out),
        str(ROOT / 'csrc/sm80/roof_o3_ring_counter_probe.cu')]
    cubin = out / (STEM + '.cubin')
    run(flags + ['-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(flags + ['-ptx', '-o', str(out / (STEM + '.ptx'))], 'ptx_build.log')
    run([str(cuda / 'cuobjdump'), '--dump-sass', str(cubin)], STEM + '.sass')
    run([str(cuda / 'cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda / 'nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    sass, ptx = (out / (STEM + '.sass')).read_text(), (out / (STEM + '.ptx')).read_text()
    entries = static_entries(sass, '^(?:' + CONTROL + '|' + SYMBOL + ')$', {CONTROL, SYMBOL})
    for symbol in (CONTROL, SYMBOL):
        block = next(b for b in re.split(r'(?=\.visible \.entry )', ptx)
                     if b.startswith('.visible .entry ' + symbol + '('))
        if not all(s in block for s in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32')):
            raise ValueError('same-entry native PTX math/copy missing')
    live = {s: analyze((out / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)}
    r.update(entries=entries, liveness=live, cost_gate=cost_gate(live[CONTROL], live[SYMBOL]),
        cubin_sha256=sha(cubin), control_comparison=compare(
            (baseline / 'o3_grouped_cta.sass').read_text(), sass, '^' + CONTROL + '$'))
    r['artifact_sha256'] = {f.name: sha(f) for f in out.iterdir()
        if f.is_file() and f.name not in ('codegen.json', 'progress.json')}
    (out / 'codegen.json').write_text(json.dumps(r, indent=2) + '\n')
    checked(out)
    print(json.dumps(dict(cost_gate=r['cost_gate'], liveness=live), indent=2), flush=True)
    print('Compile gate passed; GPU correctness/resources/full24 still required' if r['cost_gate']['passed']
          else 'Compile gate failed; stop without candidate GPU performance claims', flush=True)


if __name__ == '__main__':
    main()

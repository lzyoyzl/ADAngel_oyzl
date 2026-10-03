#!/usr/bin/env python3
"""v84: one three-stage full-K candidate, unchanged v78 math and tile."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header, SYMBOL as CONTROL
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'adangel_roof_o78_three_stage_candidate'
OLD_PIPELINE = '''  prefetch(s,0,0,a,w,af,wf,m,n,k);
  for(int group=0;group<Groups;++group) {
    const int slot=group%2;
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads(); // All prior readers finish before a slot is reused.
    if(group+1<Groups) prefetch(s,1-slot,group+1,a,w,af,wf,m,n,k);
'''
NEW_PIPELINE = '''  prefetch(s,0,0,a,w,af,wf,m,n,k);
  prefetch(s,1,1,a,w,af,wf,m,n,k);
  for(int group=0;group<Groups;++group) {
    const int slot=group%3;
    // The newest committed group may stay pending; all older groups complete.
    // At the final iteration no newer group exists, so fully drain the ring.
    if(group+1<Groups) asm volatile("cp.async.wait_group 1;" ::: "memory");
    else asm volatile("cp.async.wait_group 0;" ::: "memory");
    // Cross-thread visibility and the previous group's last readers are both
    // required before reusing (group+2)%3 == (group-1)%3.
    __syncthreads();
    if(group+2<Groups) prefetch(s,(group+2)%3,group+2,a,w,af,wf,m,n,k);
'''
SUBSTITUTIONS = (
    ('O3AmpereConfig<64,128,128,false,2,true,2>', 'O3AmpereConfig<64,128,128,false,2,true,3>'),
    ('activation_factors[2][64]', 'activation_factors[3][64]'),
    ('low[2][64*64],high[2][64*64],weight[2][128*64]', 'low[3][64*64],high[3][64*64],weight[3][128*64]'),
    ('weight_factors[2][128]', 'weight_factors[3][128]'),
    ('sizeof(Storage)==34304', 'sizeof(Storage)==51456'),
    (OLD_PIPELINE, NEW_PIPELINE),
)


def generated_header(source):
    text = eight_header(source)
    for old, new in SUBSTITUTIONS:
        if text.count(old) != 1:
            raise ValueError('v78 source boundary drift: ' + old)
        text = text.replace(old, new)
    return text.replace('o78_eight_chain_experiment', 'o78_three_stage_experiment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v78_codegen'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    prior = json.loads((args.baseline / 'codegen.json').read_text())
    for path, sha in prior['sources'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('v78 source drift: ' + path)
    if digest(args.baseline / 'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 cubin drift')
    cuda = Path('/usr/local/cuda-12.8/bin')
    cutlass = ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda / 'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    source = (ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()
    (out / 'o78_eight_chain_generated.cuh').write_text(eight_header(source))
    header = out / 'o78_three_stage_generated.cuh'
    header.write_text(generated_header(source))
    sources = set(prior['sources']) | {
        'csrc/sm80/roof_o78_three_stage_probe.cu', 'scripts/probe_o78_three_stage_codegen.py'}
    receipt = dict(scope='three_stage_compile_not_runtime_or_performance',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={s: digest(ROOT / s) for s in sorted(sources)},
        generated_header_sha256=digest(header), baseline_cubin_sha256=prior['cubin_sha256'],
        production_default_changed=False, cta_tile=[64, 128, 128], threads=128,
        integer_stages=3, unchanged_fp32_fallback_stages=2, shared_bytes=51456,
        nvcc=version, cutlass_commit=commit, commands=[])

    def run(command, filename):
        receipt['commands'].append(command)
        with (out / filename).open('w') as f:
            subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)

    compiler = [str(cuda / 'nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo', '-arch=sm_80',
        '-I' + str(cutlass / 'include'), '-I' + str(ROOT / 'csrc/sm80'), '-I' + str(out),
        str(ROOT / 'csrc/sm80/roof_o78_three_stage_probe.cu')]
    cubin = out / 'o78_three_stage.cubin'
    run(compiler + ['-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(compiler + ['-ptx', '-o', str(out / 'o78_three_stage.ptx')], 'ptx_build.log')
    run([str(cuda / 'cuobjdump'), '--dump-sass', str(cubin)], 'o78_three_stage.sass')
    run([str(cuda / 'cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda / 'nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    sass = (out / 'o78_three_stage.sass').read_text()
    entries = static_entries(sass, '^adangel_roof_o78_(?:eight_chain_candidate|three_stage_candidate)$', {CONTROL, SYMBOL})
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in entries.values()):
        raise ValueError('same-entry native INT4/async copy audit failed')
    ptx = next(b for b in re.split(r'(?=\.visible \.entry )', (out / 'o78_three_stage.ptx').read_text())
               if b.startswith('.visible .entry ' + SYMBOL + '('))
    if not all(s in ptx for s in ('cp.async.wait_group 1', 'cp.async.wait_group 0',
            'cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32')):
        raise ValueError('PTX wait/drain/native math contract failed')
    receipt.update(cubin_sha256=digest(cubin), entries=entries,
        control_comparison=compare((args.baseline / 'o78_eight_chain.sass').read_text(), sass, '^' + CONTROL + '$'),
        liveness={s: analyze((out / 'liveness.txt').read_text(), s) for s in (CONTROL, SYMBOL)},
        artifact_sha256={p: digest(out / p) for p in ('build.log', 'o78_three_stage.sass',
            'o78_three_stage.ptx', 'resources.txt', 'liveness.txt')})
    (out / 'codegen.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not receipt['control_comparison']['passed']:
        raise ValueError('included v78 control changed machine code')
    print((out / 'build.log').read_text())
    print(json.dumps(receipt['liveness'][SYMBOL], indent=2))


if __name__ == '__main__':
    main()

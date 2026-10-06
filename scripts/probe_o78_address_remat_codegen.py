#!/usr/bin/env python3
"""v91 single compile gate: materialize A cp.async addresses at use.

Motivated by v90 dynamic spill evidence. Not a tile/stage/quantizer sweep.
Retain v78/v89 encoded controls. Only positive compile evidence warrants
correctness checks and direct full24 paired timing; no small performance run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_grouped_cta_codegen import ROOT, generated_headers, checked as checked_grouped
from probe_roof_fullk_integer_codegen import static_entries

SYMBOL = 'adangel_roof_o78_address_remat_candidate'
CONTROLS = ('adangel_roof_o78_eight_chain_candidate', 'adangel_roof_o78_grouped_cta_candidate')
STEM = 'o78_address_remat'
OLD = '''    const auto* src=a+group*m*64+(roof_grouped_cta::tile().y*64+row)*64+col;
    copy16(s.low[slot]+la(row,col),src);
    copy16(s.high[slot]+la(row,col),src+m*(k/2));'''
NEW = '''    const uint32_t group_offset=group*m*64;
    // row*64 is a uint32 multiple of64 and col<=63: combining these
    // two components cannot overflow beyond their original pointer sum.
    const uint32_t row_offset=(roof_grouped_cta::tile().y*64+row)*64+col;
    roof_address_remat::copy16<false>(s.low[slot]+la(row,col),a,group_offset,row_offset,0);
    roof_address_remat::copy16<true>(s.high[slot]+la(row,col),a,group_offset,row_offset,m*(k/2));'''


def generated_header():
    source = generated_headers('o78')[0]
    if source.count(OLD) != 1:
        raise ValueError('v89 A-copy source boundary drift')
    return source.replace(OLD, NEW).replace('o78_grouped_cta_experiment', 'o78_address_remat_experiment')


def address_offsets(group, m, k, logical_y, thread, chunk):
    """CPU oracle of uint32 components; not simulated tensor-core timing."""
    if not (0 <= group < 32 and 0 <= thread < 128 and 0 <= chunk < 2
            and m > 0 and k > 0 and logical_y >= 0):
        raise ValueError('valid tile component required')
    mask = (1 << 32) - 1
    off = thread * 16 + chunk * 2048
    row, col = divmod(off, 64)
    g = (group * m * 64) & mask
    r = (((logical_y * 64 + row) & mask) * 64) & mask
    high = (m * (k // 2)) & mask
    return (g + r + col, g + r + col + high), (g + ((r + col) & mask), g + ((r + col) & mask) + high)


def opcode_count(loop, opcode):
    """nvdisasm retains width/type suffixes, for example LDL.64."""
    return sum(count for name, count in loop['opcode_counts'].items()
               if name.split('.')[0] == opcode)


def runtime_justified(candidate, prior):
    new_loop = next(loop for loop in candidate['loops'] if loop['kind'] == 'integer')
    old_loop = next(loop for loop in prior['loops'] if loop['kind'] == 'integer')
    return (opcode_count(new_loop, 'LDL') < opcode_count(old_loop, 'LDL')
            and candidate['allocated_gpr'] <= 168)


def checked(directory):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    r = json.loads((directory / 'codegen.json').read_text())
    for path, digest in r['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('rematerialization source drift: ' + path)
    for path, digest in r['artifact_sha256'].items():
        if sha(directory / path) != digest:
            raise ValueError('rematerialization artifact drift: ' + path)
    if (directory / (STEM + '_generated.cuh')).read_text() != generated_header():
        raise ValueError('rematerialized header drift')
    if not all(value['passed'] for value in r['control_comparisons'].values()):
        raise ValueError('encoded control changed')
    if r['changed_semantics'] or r['production_default_changed']:
        raise ValueError('math or default changed')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in r['entries'].values()):
        raise ValueError('native math/copy gate failed')
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v89_o78_codegen'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    prior = checked_grouped(args.baseline, 'o78')
    cuda = Path('/usr/local/cuda-12.8/bin')
    cutlass = ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda / 'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    for name in ('o78_eight_chain_generated.cuh', 'o78_grouped_cta_generated.cuh', 'o78_grouped_fallback_generated.cuh'):
        (out / name).write_text((args.baseline / name).read_text())
    (out / (STEM + '_generated.cuh')).write_text(generated_header())
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    sources = set(prior['sources']) | {'csrc/sm80/roof_cp_async_offset.cuh',
        'csrc/sm80/roof_o78_address_remat_probe.cu', 'scripts/probe_o78_address_remat_codegen.py'}
    receipt = dict(scope='v91_address_rematerialization_compile_gate_only',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={s: sha(ROOT / s) for s in sorted(sources)}, nvcc=version, cutlass_commit=commit,
        baseline_cubin_sha256=prior['cubin_sha256'], cta_tile=[64, 128, 128], threads=128,
        shared_bytes=34304, stages=2, changed_semantics=False, production_default_changed=False,
        candidate=SYMBOL, controls=list(CONTROLS), commands=[])
    def run(cmd, name):
        receipt['commands'].append(cmd)
        with (out / name).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    flags = [str(cuda / 'nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
        '-arch=sm_80', '-DADANGEL_FULLK_INTEGER=1', '-I' + str(cutlass / 'include'),
        '-I' + str(ROOT / 'csrc/sm80'), '-I' + str(out), str(ROOT / 'csrc/sm80/roof_o78_address_remat_probe.cu')]
    cubin = out / (STEM + '.cubin')
    run(flags + ['-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(flags + ['-ptx', '-o', str(out / (STEM + '.ptx'))], 'ptx_build.log')
    run([str(cuda / 'cuobjdump'), '--dump-sass', str(cubin)], STEM + '.sass')
    run([str(cuda / 'cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda / 'nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    sass = (out / (STEM + '.sass')).read_text()
    ptx = (out / (STEM + '.ptx')).read_text()
    symbols = {*CONTROLS, SYMBOL}
    entries = static_entries(sass, '^(?:' + '|'.join(sorted(symbols)) + ')$', symbols)
    for symbol in symbols:
        body = next(b for b in re.split(r'(?=\.visible \.entry )', ptx) if b.startswith('.visible .entry ' + symbol + '('))
        if not all(x in body for x in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32')):
            raise ValueError('same-entry PTX math/copy gate failed')
    live = (out / 'liveness.txt').read_text()
    receipt.update(entries=entries, liveness={symbol: analyze(live, symbol) for symbol in sorted(symbols)},
        cubin_sha256=sha(cubin), control_comparisons={symbol: compare(
            (args.baseline / 'o78_grouped_cta.sass').read_text(), sass, '^' + symbol + '$') for symbol in CONTROLS})
    receipt['worth_runtime_validation'] = runtime_justified(
        receipt['liveness'][SYMBOL], receipt['liveness'][CONTROLS[1]])
    receipt['artifact_sha256'] = {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.name != 'codegen.json'}
    (out / 'codegen.json').write_text(json.dumps(receipt, indent=2) + '\n')
    checked(out)
    print(json.dumps({symbol: receipt['liveness'][symbol] for symbol in sorted(symbols)}, indent=2), flush=True)
    print('Compile audit passed; runtime justified:', receipt['worth_runtime_validation'], flush=True)


if __name__ == '__main__':
    main()

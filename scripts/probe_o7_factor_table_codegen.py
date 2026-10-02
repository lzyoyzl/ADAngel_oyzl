#!/usr/bin/env python3
"""v76: move repeated O7 coefficient multiplies to a CTA shared table.

Same G128 semantics, full-K guard and two native INT4 routes; do not enable in
production. First gate is code/resource cost, before any runtime claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_fullk_candidate'
CANDIDATE = 'adangel_roof_o7_factor_table_candidate'
TABLE_ROWS = 10
SHARED_BYTES = 43520


def table_header(source):
    """Mechanical, checked differences from the audited v67 body.

    Generated source is saved with the binary and hashed. No editing of the
    original candidate or production headers occurs.
    """
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('v67 source pattern drift: ' + old[:60])
        source = source.replace(old, new)
    source = source.replace('o78_fullk_integer_experiment', 'o7_factor_table_experiment')
    replace('  int weight_factors[2][128];', '  uint32_t coefficients[2][10][128];')
    replace('static_assert(sizeof(Storage)==34304);\nstatic_assert(sizeof(Storage)==sizeof(C::Storage));',
            'static_assert(sizeof(Storage)==43520);')
    replace('''  // Factors share the old scale-panel layout, commit/wait and CTA barrier.
  const unsigned first=threadIdx.x*4;
  if(threadIdx.x<16) copy16(s.activation_factors[slot]+first,
      af+group*m+blockIdx.y*64+first);
  if(threadIdx.x<32) copy16(s.weight_factors[slot]+first,
      wf+group*n+blockIdx.x*128+first);''', '''  // Exact in-CTA table: one column per producer thread. Unsigned shifts are
  // defined for unused entries too; the old coefficient guard proves that
  // every actually selected entry is <=INT32_MAX. No quantization change.
  const uint32_t factor=static_cast<uint32_t>(wf[group*n+blockIdx.x*128+threadIdx.x]);
  o1_static_for<0,10>([&](auto shift) {
    s.coefficients[slot][shift][threadIdx.x]=factor<<shift;
  });
  if(threadIdx.x<64) {
    const uint32_t row_factor=static_cast<uint32_t>(af[group*m+blockIdx.y*64+threadIdx.x]);
    // The entry guard has proven row_factor is a positive in-table power of2.
    s.activation_factors[slot][threadIdx.x]=31-__clz(row_factor);
  }
  // Ordinary shared stores are ordered by the same next-stage CTA barrier.
  // No extra per-G128 barrier; cp.async continues to carry all A/W payload.''')
    replace('''            const int coefficient=s.activation_factors[slot][cute::get<0>(coord)]*
                                  s.weight_factors[slot][cute::get<1>(coord)];''', '''            const int shift=s.activation_factors[slot][cute::get<0>(coord)];
            const int coefficient=static_cast<int>(s.coefficients[slot][shift][cute::get<1>(coord)]);''')
    return '// Generated v76 table path from unchanged v67, not a formal backend.\n' + source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error('fresh repository output required')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    original = json.loads((a.baseline / 'codegen.json').read_text())
    for name, sha in original['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('v67 source drift: ' + name)
    if digest(a.baseline / 'o78_fullk.cubin') != original['cubin_sha256']:
        raise ValueError('v67 binary drift')
    cuda, cutlass = Path('/usr/local/cuda-12.8'), ROOT / 'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda / 'bin/nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        p.error('pinned CUDA/CUTLASS required')
    out.mkdir(parents=True)
    header = out / 'o7_factor_table_generated.cuh'
    header.write_text(table_header((ROOT / 'csrc/sm80/o78_fullk_integer_probe.cuh').read_text()))
    names = set(original['sources']) | {'csrc/sm80/roof_o7_factor_table_probe.cu',
                                      'scripts/probe_o7_factor_table_codegen.py'}
    receipt = dict(scope='isolated_compile_gate_not_runtime_performance_or_MSE',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={name: digest(ROOT / name) for name in sorted(names)},
        generated_header_sha256=digest(header), nvcc=version, cutlass_commit=commit,
        control_original_cubin_sha256=original['cubin_sha256'], shared_bytes=SHARED_BYTES,
        table_rows=TABLE_ROWS, exact_factor_guard='power_of_two_positive_and_less_than_1024',
        default_changed=False, commands=[])
    def run(cmd, filename):
        receipt['commands'].append(cmd)
        with (out / filename).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    base = [str(cuda / 'bin/nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
            '-arch=sm_80', '-I'+str(cutlass / 'include'), '-I'+str(ROOT / 'csrc/sm80'), '-I'+str(out),
            str(ROOT / 'csrc/sm80/roof_o7_factor_table_probe.cu')]
    cubin = out / 'o7_factor_table.cubin'
    run(base + ['-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(base + ['-ptx', '-o', str(out / 'o7_factor_table.ptx')], 'ptx_build.log')
    run([str(cuda / 'bin/cuobjdump'), '--dump-sass', str(cubin)], 'o7_factor_table.sass')
    run([str(cuda / 'bin/cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda / 'bin/nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    symbols = {CONTROL, CANDIDATE, 'adangel_roof_o78_fullk_control'}
    entries = static_entries((out / 'o7_factor_table.sass').read_text(),
                            r'^adangel_roof_o(?:78_fullk_(?:candidate|control)|7_factor_table_candidate)$', symbols)
    for symbol, entry in entries.items():
        assert entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma'] and entry['all_copies_bypass_l1'], symbol
    ptx = (out / 'o7_factor_table.ptx').read_text()
    for symbol in symbols:
        body = next(b for b in re.split(r'(?=\.visible \.entry )', ptx) if b.startswith('.visible .entry ' + symbol + '('))
        assert all(op in body for op in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32'))
    old = original['entries'][CONTROL]
    receipt.update(entries=entries, cubin_sha256=digest(cubin),
        control_opcode_counts_match_v67=entries[CONTROL]['opcode_counts'] == old['opcode_counts'],
        control_instruction_count_match_v67=entries[CONTROL]['instructions'] == old['instructions'])
    (out / 'codegen.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print((out / 'resources.txt').read_text())
    print('native INT4 audit passed; inspect table-loop loads/integer work and spill before runtime')


if __name__ == '__main__':
    main()

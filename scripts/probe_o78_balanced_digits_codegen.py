#!/usr/bin/env python3
"""v100 feasibility gate, not a new production backend or performance result.

O7 local fixed integers are in [-112,112], O8 in [-30,30]. Their exact radix16
digits can both be signed INT4: lo=signed4(q&15), hi=floor((q+8)/16).
The low nibble bits stay unchanged; the high nibble MUST be recoded during
online conversion. Both the guarded full-K body and FP32 fallback use S4/S4.

This probes whether the single MMA type lets ptxas materially reduce register
pressure or hot-loop work. Merely changing U4 to S4 is NOT presumed faster.
No GPU timing is justified by identical instruction/resource budgets.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

ROOT = Path(__file__).resolve().parents[1]
CONTROL = 'adangel_roof_o78_eight_chain_candidate'
SYMBOL = 'adangel_roof_o78_balanced_digits_candidate'
BASELINE = 'reports/o378_roof_v78_codegen'
ATOM_OLD = 'SM80_16x8x64_S32U4S4S32_TN'
ATOM_NEW = 'SM80_16x8x64_S32S4S4S32_TN'


def balanced_pair(q):
    """Exact digits, rejecting INT8 values whose compensated high exceeds S4."""
    if not isinstance(q, int) or not -128 <= q <= 119:
        raise ValueError('balanced INT4 digits require -128 <= q <= 119')
    hi = (q + 8) // 16
    lo = q - 16 * hi
    if not (-8 <= lo <= 7 and -8 <= hi <= 7 and lo + 16 * hi == q):
        raise AssertionError('radix16 reconstruction failed')
    return lo, hi


def scalar_proof():
    path = ROOT/'python/adangel/quantization/mixed_formats.py'
    spec = importlib.util.spec_from_file_location('balanced_scalar_formats', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = {}
    for variant, kind, bits, f, bound in (
            ('o7', 'e4m3', 8, -2, 112), ('o8', 'e2m3', 6, 2, 30)):
        count, integers, highs = 0, [], []
        for code in range(256 if kind == 'e4m3' else 64):
            if kind == 'e4m3' and code in (127, 255):
                try:
                    module.decode_scalar(code, kind)
                except ValueError:
                    continue
                raise AssertionError('NaN code was silently accepted')
            value = module.decode_scalar(code, kind)
            q = module.fixed_scalar(value, bits, f)
            lo, hi = balanced_pair(q)
            # Exact expression used by the generated CUDA converter, including
            # negative integers interpreted as unsigned32 before shifting.
            packed_hi = (((q + 8) & 0xffffffff) >> 4) & 15
            packed_lo = q & 15
            signed4 = lambda x: x - 16 if x & 8 else x
            assert signed4(packed_lo) == lo and signed4(packed_hi) == hi
            assert abs(q) <= bound
            integers.append(q); highs.append(hi); count += 1
        records[variant] = dict(valid_source_codes=count, q_min=min(integers), q_max=max(integers),
                                high_min=min(highs), high_max=max(highs), fractional_bits=f)
    # Test every representable balanced digit input, not only source codebook.
    for q in range(-128, 120):
        lo, hi = balanced_pair(q)
        assert lo + 16 * hi == q
    return dict(exact_integer_reconstruction=True, source_codes=records,
                exhaustive_integer_inputs=248, output_or_MSE_runtime_tested=False,
                quantization_scale_guard_math_changed=False,
                o3_excluded=True, unsupported_int8_inputs=[120,127])


def generated_headers():
    fallback = (ROOT/'csrc/sm80/o78_unsigned_payload_candidate.cuh').read_text()
    fullk = eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
    converter = (ROOT/'csrc/sm80/roof_row_fused_conversion.cuh').read_text()
    if 'cutlass::uint4b_t' not in fallback or ATOM_OLD not in fallback:
        raise ValueError('fallback low-MMA source drift')
    fallback = fallback.replace('o78_unsigned_payload_experiment', 'o78_balanced_payload_experiment')
    fallback = fallback.replace('cutlass::uint4b_t', 'cutlass::int4b_t').replace(ATOM_OLD, ATOM_NEW)
    if 'O78::' not in fullk or ATOM_OLD not in fullk:
        raise ValueError('integer low-MMA source drift')
    fullk = fullk.replace('o78_eight_chain_experiment', 'o78_balanced_integer_experiment')
    fullk = fullk.replace('O78::', 'O78Balanced::')
    fullk = fullk.replace('cutlass::uint4b_t', 'cutlass::int4b_t').replace(ATOM_OLD, ATOM_NEW)
    old = 'if constexpr(!Weight) high[j/8]|=((unsigned(q)>>4)&15u)<<(4*(j%8));'
    new = 'if constexpr(!Weight) high[j/8]|=((unsigned(q+8)>>4)&15u)<<(4*(j%8));'
    if converter.count(old) != 1:
        raise ValueError('row-fused high-packing source drift')
    converter = converter.replace('row_fused_probe', 'balanced_row_fused_probe').replace(old, new)
    return dict(o78_balanced_payload_generated_cuh=fallback,
                o78_balanced_integer_generated_cuh=fullk,
                o78_balanced_conversion_generated_cuh=converter)


def runtime_gate(control, candidate):
    old = next(x for x in control['loops'] if x['kind'] == 'integer')
    new = next(x for x in candidate['loops'] if x['kind'] == 'integer')
    local = lambda loop: sum(v for op, v in loop['opcode_counts'].items() if op.split('.')[0] in ('LDL', 'STL'))
    counts = new['opcode_counts']
    capacity = candidate['allocated_gpr'] <= 128 < control['allocated_gpr']
    lower_work = new['static_instructions'] <= old['static_instructions'] - 8
    budget = (candidate['allocated_gpr'] <= control['allocated_gpr']
              and new['max_live_gpr'] <= old['max_live_gpr'] and local(new) <= local(old)
              and new['static_instructions'] <= old['static_instructions']
              and counts.get('IMMA.16864.S4.S4') == 64
              and not counts.get('IMMA.16864.U4.S4', 0)
              and counts.get('LDSM.16.M88.4') == 16)
    return dict(worth_runtime_validation=budget and (capacity or lower_work),
                no_extra_hot_work_budget=budget, register_capacity_improved=capacity,
                integer_instructions_saved=old['static_instructions']-new['static_instructions'],
                minimum_instruction_saving_required=8, preparation_cost_measured=False,
                interpretation='compiler_screen_only_no_measured_speedup_or_output_MSE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(BASELINE))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); baseline = args.baseline.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    prior = json.loads((baseline/'codegen.json').read_text())
    for name, digest in prior['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('v78 source drift: '+name)
    if sha(baseline/'o78_eight_chain.cubin') != prior['cubin_sha256']:
        raise ValueError('v78 binary drift')
    cuda, cutlass = Path('/usr/local/cuda-12.8/bin'), ROOT/'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git','-C',str(cutlass),'rev-parse','HEAD'],text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA12.8/CUTLASS required')
    proof = scalar_proof(); out.mkdir(parents=True)
    for file in baseline.iterdir():
        if file.suffix in ('.cu', '.cuh'):
            (out/file.name).write_text(file.read_text())
    generated = generated_headers()
    for name, text in generated.items():
        (out/name.replace('_cuh','.cuh')).write_text(text)
    sources = set(prior['sources']) | {'scripts/probe_o78_balanced_digits_codegen.py',
        'csrc/sm80/roof_o78_balanced_digits_probe.cu', 'csrc/sm80/roof_row_fused_conversion.cuh',
        'python/adangel/quantization/mixed_formats.py'}
    receipt = dict(scope='v100_exact_balanced_digits_compile_gate_not_runtime',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources={name:sha(ROOT/name) for name in sorted(sources)}, nvcc=version, cutlass_commit=commit,
        scalar_proof=proof, cta_tile=[64,128,128], threads=128, stages=2, shared_bytes=34304,
        logical_partial_registers=32, source_max_chains=8, production_default_changed=False,
        o3_changed=False, o78_quantization_changed=False, activation_packing_changed=True,
        numerical_GPU_validation_run=False, performance_measurement_run=False,
        baseline_cubin_sha256=prior['cubin_sha256'], commands=[])
    def run(command, name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    flags = [str(cuda/'nvcc'),'-O3','-std=c++17','--expt-relaxed-constexpr','-lineinfo','-arch=sm_80',
        '-I'+str(cutlass/'include'),'-I'+str(ROOT/'csrc/sm80'),'-I'+str(out),
        str(ROOT/'csrc/sm80/roof_o78_balanced_digits_probe.cu')]
    cubin = out/'o78_balanced_digits.cubin'
    run(flags+['-cubin','-o',str(cubin),'-Xptxas=-v'],'build.log')
    run(flags+['-ptx','-o',str(out/'o78_balanced_digits.ptx')],'ptx_build.log')
    run([str(cuda/'cuobjdump'),'--dump-sass',str(cubin)],'o78_balanced_digits.sass')
    run([str(cuda/'cuobjdump'),'--dump-resource-usage',str(cubin)],'resources.txt')
    run([str(cuda/'nvdisasm'),'--print-code','--life-range-mode','count',str(cubin)],'liveness.txt')
    sass, ptx = (out/'o78_balanced_digits.sass').read_text(), (out/'o78_balanced_digits.ptx').read_text()
    symbols = {CONTROL,SYMBOL}
    entries = static_entries(sass, '^(?:'+'|'.join(sorted(symbols))+')$', symbols)
    for symbol, entry in entries.items():
        body = next(b for b in re.split(r'(?=\.visible \.entry )',ptx)
                    if b.startswith('.visible .entry '+symbol+'('))
        assert 'cp.async.cg.shared.global' in body and '.s32.s4.s4.s32' in body
        assert entry['native_s4_s4'] and not entry['int8_mma'] and entry['all_copies_bypass_l1']
        if symbol == SYMBOL:
            assert '.s32.u4.s4.s32' not in body and not entry['native_u4_s4']
        else:
            assert '.s32.u4.s4.s32' in body and entry['native_u4_s4']
    lives = {symbol:analyze((out/'liveness.txt').read_text(),symbol) for symbol in symbols}
    receipt.update(entries=entries,liveness=lives,cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/'o78_eight_chain.sass').read_text(),sass,'^'+CONTROL+'$'),
        runtime_gate=runtime_gate(lives[CONTROL],lives[SYMBOL]))
    if not receipt['control_comparison']['passed']:
        raise ValueError('old v78 machine instructions changed')
    receipt['artifact_sha256'] = {f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name != 'codegen.json'}
    (out/'codegen.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(scalar_proof=proof, runtime_gate=receipt['runtime_gate'],liveness=lives),indent=2),flush=True)


if __name__ == '__main__':
    main()

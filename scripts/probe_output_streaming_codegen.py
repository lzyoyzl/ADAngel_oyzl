#!/usr/bin/env python3
"""v99: one streaming-output policy, no input-cache/tile/stage sweep.

Change only the guarded integer epilogue's existing stores to __stcs. The
fallback, payload, factors, eight chains and all arithmetic remain unchanged.
Old v37/v38 changed INPUT cache/prefetch; they did not test these output stores.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_interleaved_tail_codegen import ROOT, CONFIG as OLD
from probe_grouped_cta_codegen import generated_headers, checked as grouped_checked
from probe_o78_eight_chain_codegen import generated_header as eight_header
from probe_roof_fullk_integer_codegen import static_entries

CONFIG = {kind: dict(cfg, symbol=f'adangel_roof_{kind}_output_streaming_candidate',
                    stem=f'{kind}_output_streaming') for kind, cfg in OLD.items()}
REFERENCE = 'https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#cache-operators'


def generated_header(kind):
    if kind == 'o3':
        text = generated_headers('o3')[0]
        old_namespace = 'o3_grouped_cta_experiment'
    elif kind == 'o78':
        text = eight_header((ROOT/'csrc/sm80/o78_fullk_integer_probe.cuh').read_text())
        old_namespace = 'o78_eight_chain_experiment'
    else:
        raise ValueError('O3 or O7/O8 required')
    vector = '*reinterpret_cast<float2*>(y+offset)=make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{})));'
    if text.count(vector) != 1:
        raise ValueError('existing vector-store boundary drift')
    text = text.replace(vector, '__stcs(reinterpret_cast<float2*>(y+offset),make_float2(__int_as_float(acc(i)),__int_as_float(acc(i+cute::_1{}))));')
    scalar = 'y[offset]=__int_as_float(acc(i));'
    if text.count(scalar) != 1:
        raise ValueError('existing scalar-store boundary drift')
    text = text.replace(scalar, '__stcs(y+offset,__int_as_float(acc(i)));')
    tail = re.compile(r'y\[(.*?)\]=__int_as_float\(acc\(i\+cute::_1\{\}\)\);')
    text, count = tail.subn(r'__stcs(y+(\1),__int_as_float(acc(i+cute::_1{})));', text)
    if count != 1 or text.count('__stcs(') != 3:
        raise ValueError('existing scalar-tail boundary drift')
    return text.replace(old_namespace, kind+'_output_streaming_experiment')


def prior_memory_evidence(kind):
    path = ROOT/'docs/evidence/a100_o378_roof_v90_ncu/reports/o378_roof_v90_ncu'/(
        'o3_1_source_sass.csv' if kind == 'o3' else 'o7_0_source_sass.csv')
    with path.open(newline='') as source:
        next(source)  # kernel identity record, not the CSV column header
        rows = list(csv.DictReader(source))
    result = dict(source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  kernel=CONFIG[kind]['control'], executed_ldsm=0, ldsm_excess_wavefronts=0,
                  executed_stg=0, output_streaming_already_present=False)
    for row in rows:
        instruction = row['Source'].strip()
        executed = int(row['Instructions Executed'])
        if re.search(r'\bLDSM\.', instruction):
            result['executed_ldsm'] += executed
            result['ldsm_excess_wavefronts'] += int(row['L1 Wavefronts Shared Excessive'])
        if re.search(r'\bSTG\.', instruction) and executed:
            result['executed_stg'] += executed
            result['output_streaming_already_present'] |= bool(re.search(r'\bSTG\.\S*(?:CS|EF)', instruction))
    if result['executed_ldsm'] != 4194304 or result['ldsm_excess_wavefronts'] or result['output_streaming_already_present']:
        raise ValueError('authoritative source-memory premise changed')
    return result


def store_audit(sass, symbol):
    block = next(b for b in re.split(r'(?=\s*Function\s*:\s*)', sass)
                 if re.match(r'\s*Function\s*:\s*'+re.escape(symbol)+r'\s', b))
    stores = Counter()
    for opcode in re.findall(r'\bSTG(?:\.[A-Z0-9]+)+', block):
        stores[opcode] += 1
    return dict(opcodes=dict(sorted(stores.items())), total=sum(stores.values()),
                streaming=sum(v for op, v in stores.items() if 'EF' in op.split('.') or 'CS' in op.split('.')))


def worth_runtime(control, candidate, stores):
    old = next(x for x in control['loops'] if x['kind'] == 'integer')
    new = next(x for x in candidate['loops'] if x['kind'] == 'integer')
    counts = new['opcode_counts']
    local = lambda loop: sum(v for op, v in loop['opcode_counts'].items() if op.split('.')[0] in ('LDL', 'STL'))
    return (candidate['allocated_gpr'] <= control['allocated_gpr'] <= 168
            and new['max_live_gpr'] <= old['max_live_gpr']
            and new['static_instructions'] <= old['static_instructions']
            and local(new) <= local(old)
            and counts.get('IMMA.16864.S4.S4') == counts.get('IMMA.16864.U4.S4') == 32
            and counts.get('LDSM.16.M88.4') == 16
            and stores['candidate']['total'] == stores['control']['total']
            and stores['candidate']['streaming'] > 0 and stores['control']['streaming'] == 0)


def checked(directory, kind):
    cfg = CONFIG[kind]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((directory/'codegen.json').read_text())
    for name, digest in receipt['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('output-streaming source drift: '+name)
    for name, digest in receipt['artifact_sha256'].items():
        if sha(directory/name) != digest:
            raise ValueError('output-streaming artifact drift: '+name)
    if (directory/(cfg['stem']+'_generated.cuh')).read_text() != generated_header(kind):
        raise ValueError('generated store-policy body drift')
    if not receipt['control_comparison']['passed'] or receipt['changed_semantics'] or receipt['production_default_changed']:
        raise ValueError('control/math/default drift')
    if not all(e['native_u4_s4'] and e['native_s4_s4'] and not e['int8_mma']
               and e['all_copies_bypass_l1'] for e in receipt['entries'].values()):
        raise ValueError('same-entry native INT4/cg copy failed')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=CONFIG, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg, out = CONFIG[args.kind], args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        parser.error('fresh repository output required')
    baseline = ROOT/cfg['baseline']
    if args.kind == 'o3':
        grouped_checked(baseline, 'o3')
    prior = json.loads((baseline/'codegen.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, digest in prior['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('best source drift: '+name)
    if sha(baseline/(cfg['old_stem']+'.cubin')) != prior['cubin_sha256']:
        raise ValueError('best cubin drift')
    cuda, cutlass = Path('/usr/local/cuda-12.8/bin'), ROOT/'third_party/cutlass-src'
    version = subprocess.check_output([str(cuda/'nvcc'), '--version'], text=True)
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if 'release 12.8' not in version or commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('pinned CUDA12.8/CUTLASS required')
    premise = prior_memory_evidence(args.kind)
    out.mkdir(parents=True)
    for file in baseline.iterdir():
        if file.suffix in ('.cu', '.cuh'):
            (out/file.name).write_text(file.read_text())
    (out/(cfg['stem']+'_generated.cuh')).write_text(generated_header(args.kind))
    sources = set(prior['sources']) | {'scripts/probe_output_streaming_codegen.py',
        f"csrc/sm80/roof_{args.kind}_output_streaming_probe.cu", premise['source']}
    receipt = dict(scope='v99_fixed_output_streaming_compile_gate', kind=args.kind,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        sources={s: sha(ROOT/s) for s in sorted(sources)}, commands=[], reference=REFERENCE,
        prior_source_memory=premise, nvcc=version, cutlass_commit=commit,
        cta_tile=[64,128,128], threads=128, stages=cfg['stages'], shared_bytes=cfg['shared'],
        group_m=cfg['group_m'], partial_registers=32, source_max_chains=8,
        integer_output_policy='streaming_cs', fallback_output_policy='unchanged_default_wb',
        changed_semantics=False, production_default_changed=False,
        baseline_cubin_sha256=prior['cubin_sha256'])
    def run(command, name):
        receipt['commands'].append(command)
        with (out/name).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    flags = [str(cuda/'nvcc'), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo', '-arch=sm_80',
        '-DADANGEL_FULLK_INTEGER=1', '-I'+str(cutlass/'include'), '-I'+str(ROOT/'csrc/sm80'), '-I'+str(out),
        str(ROOT/f"csrc/sm80/roof_{args.kind}_output_streaming_probe.cu")]
    cubin = out/(cfg['stem']+'.cubin')
    run(flags+['-cubin', '-o', str(cubin), '-Xptxas=-v'], 'build.log')
    run(flags+['-ptx', '-o', str(out/(cfg['stem']+'.ptx'))], 'ptx_build.log')
    run([str(cuda/'cuobjdump'), '--dump-sass', str(cubin)], cfg['stem']+'.sass')
    run([str(cuda/'cuobjdump'), '--dump-resource-usage', str(cubin)], 'resources.txt')
    run([str(cuda/'nvdisasm'), '--print-code', '--life-range-mode', 'count', str(cubin)], 'liveness.txt')
    sass, ptx = (out/(cfg['stem']+'.sass')).read_text(), (out/(cfg['stem']+'.ptx')).read_text()
    symbols = {cfg['control'], cfg['symbol']}
    entries = static_entries(sass, '^(?:'+'|'.join(sorted(symbols))+')$', symbols)
    for symbol in symbols:
        block = next(b for b in re.split(r'(?=\.visible \.entry )', ptx)
                     if b.startswith('.visible .entry '+symbol+'('))
        if not all(token in block for token in ('cp.async.cg.shared.global', '.s32.u4.s4.s32', '.s32.s4.s4.s32')):
            raise ValueError('same-entry native INT4/cg missing')
        if symbol == cfg['symbol'] and 'st.global.cs' not in block:
            raise ValueError('streaming output absent from candidate PTX')
    lives = {s: analyze((out/'liveness.txt').read_text(), s) for s in symbols}
    stores = {label: store_audit(sass, cfg['control'] if label == 'control' else cfg['symbol'])
              for label in ('control', 'candidate')}
    receipt.update(entries=entries, liveness=lives, stores=stores, cubin_sha256=sha(cubin),
        control_comparison=compare((baseline/(cfg['old_stem']+'.sass')).read_text(), sass, '^'+cfg['control']+'$'))
    receipt['worth_runtime_validation'] = worth_runtime(lives[cfg['control']], lives[cfg['symbol']], stores)
    receipt['artifact_sha256'] = {f.name: sha(f) for f in out.iterdir() if f.is_file() and f.name != 'codegen.json'}
    (out/'codegen.json').write_text(json.dumps(receipt, indent=2)+'\n')
    checked(out, args.kind)
    print(json.dumps(dict(runtime_justified=receipt['worth_runtime_validation'], stores=stores, liveness=lives), indent=2), flush=True)


if __name__ == '__main__':
    main()

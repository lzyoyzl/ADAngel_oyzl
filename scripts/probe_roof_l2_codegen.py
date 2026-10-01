#!/usr/bin/env python3
"""Compile isolated SM80 L2-hint probes before spending GPU benchmark time.

No extension rebuild, runtime dispatch change, or numerical claim is made.
Only a fresh output directory is allowed; all compiler artifacts stay there.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from compare_a100_codegen import compare

ROOT = Path(__file__).resolve().parents[1]
PATTERN = r'^adangel_roof_l2_o(?:3|78)$'


def static_entries(sass):
    entries = {}
    for block in re.split(r'(?=Function\s*:\s*)', sass):
        if not block.startswith('Function'):
            continue
        symbol = block.splitlines()[0].split(':', 1)[1].strip()
        if not re.fullmatch(PATTERN, symbol):
            continue
        if symbol in entries:
            raise ValueError('duplicate probe entry')
        text = []
        for line in block.splitlines():
            if re.match(r'\s*/\*[0-9a-f]+\*/', line):
                ins = re.sub(r'/\*.*?\*/', '', line).strip()
                if ins:
                    text.append(ins)
        if not text:
            raise ValueError('missing instructions')
        copies = [ins for ins in text if re.search(r'\bLDGSTS\b', ins)]
        entries[symbol] = dict(instructions=len(text), copies=copies,
            all_copies_bypass_l1=bool(copies) and all('.BYPASS' in s for s in copies),
            native_u4_s4=any(re.search(r'\bIMMA[^;]*\.U4\.S4', s) for s in text),
            native_s4_s4=any(re.search(r'\bIMMA[^;]*\.S4\.S4', s) for s in text),
            int8_mma=any(re.search(r'\bIMMA[^;]*\.[SU]8\.', s) for s in text))
    if set(entries) != {'adangel_roof_l2_o3', 'adangel_roof_l2_o78'}:
        raise ValueError('missing probe entry')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda-12.8'))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error('output must be a fresh directory inside this repository')
    nvcc = args.cuda / 'bin/nvcc'
    dump = args.cuda / 'bin/cuobjdump'
    version = subprocess.check_output([str(nvcc), '--version'], text=True)
    if 'release 12.8' not in version:
        parser.error('authoritative probe requires CUDA 12.8')
    cutlass = ROOT / 'third_party/cutlass-src'
    commit = subprocess.check_output(['git', '-C', str(cutlass), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != 'db1c288993354c88e551c40c19a8fb93a774a241':
        parser.error('wrong CUTLASS revision')
    output.mkdir(parents=True)
    commands = []
    def run(command, file):
        commands.append(command)
        with (output / file).open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        nvcc=version, cutlass_commit=commit,
        scope='compile_and_static_SASS_only_not_runtime_correctness_or_performance', variants={})
    for size in (0, 128, 256):
        prefix = f'l2_{size}'
        command = [str(nvcc), '-O3', '-std=c++17', '--expt-relaxed-constexpr', '-lineinfo',
            '-arch=sm_80', f'-DADANGEL_L2_PREFETCH_BYTES={size}', '-I' + str(cutlass / 'include'),
            '-I' + str(cutlass / 'tools/util/include'), 'csrc/sm80/roof_l2_prefetch_probe.cu']
        cubin = output / f'{prefix}.cubin'
        run(command + ['-cubin', '-o', str(cubin), '-Xptxas=-v'], f'{prefix}_build.log')
        run(command + ['-ptx', '-o', str(output / f'{prefix}.ptx')], f'{prefix}_ptx_build.log')
        run([str(dump), '--dump-sass', str(cubin)], f'{prefix}.sass')
        run([str(dump), '--dump-resource-usage', str(cubin)], f'{prefix}_resources.txt')
        sass = (output / f'{prefix}.sass').read_text()
        manifest['variants'][str(size)] = dict(cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
            entries=static_entries(sass))
        if size:
            manifest['variants'][str(size)]['encoded_comparison_to_no_hint'] = compare(
                (output / 'l2_0.sass').read_text(), sass, PATTERN)
    manifest['commands'] = commands
    manifest['all_probe_copies_bypass_l1'] = all(
        entry['all_copies_bypass_l1'] for row in manifest['variants'].values() for entry in row['entries'].values())
    manifest['native_int4_entries'] = all(entry['native_u4_s4'] and entry['native_s4_s4'] and not entry['int8_mma']
        for row in manifest['variants'].values() for entry in row['entries'].values())
    (output / 'codegen.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
    if not manifest['native_int4_entries'] or not manifest['all_probe_copies_bypass_l1']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

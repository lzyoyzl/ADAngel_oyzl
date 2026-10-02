#!/usr/bin/env python3
"""Read-only v75 diagnosis of the existing v67 cubin, not a new kernel.

nvdisasm reports static live ranges for this binary; this is not a dynamic
register-pressure profile or a proof that every equivalent algorithm needs
these registers. Natural backward branches identify the two long K loops.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = 'adangel_roof_o78_fullk_candidate'


def analyze(text, symbol=SYMBOL):
    functions = re.split(r'(?=^//-+ \.text\.)', text, flags=re.M)
    block = next((b for b in functions if re.match(r'//-+ \.text\.' + re.escape(symbol) + r'\s', b)), None)
    if block is None:
        raise ValueError('exact kernel section missing')
    allocated = re.search(r'SHI_REGISTERS=(\d+)', block)
    if not allocated:
        raise ValueError('allocated register metadata missing')
    instructions, labels, pending = [], {}, []
    for line in block.splitlines():
        label = re.match(r'\s*(\.L_[A-Za-z0-9_]+):', line)
        if label:
            pending.append(label[1])
        ins = re.search(r'/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*// \|\s*(\d+)\s*\|', line)
        if ins:
            pc, op, live = int(ins[1], 16), ins[2].strip(), int(ins[3])
            for name in pending:
                labels[name] = pc
            pending.clear()
            instructions.append(dict(pc=pc, instruction=op, live_gpr=live))
    if not instructions:
        raise ValueError('no instruction-level count-mode liveness found')
    loops = []
    for ins in instructions:
        branch = re.search(r'\bBRA\s+`\((\.L_[A-Za-z0-9_]+)\)', ins['instruction'])
        if branch and branch[1] not in labels:
            raise ValueError('unresolved branch target')
        if not branch or labels[branch[1]] >= ins['pc']:
            continue
        region = [i for i in instructions if labels[branch[1]] <= i['pc'] <= ins['pc']]
        counts = Counter(re.sub(r'^@!?[A-Z0-9]+\s+', '', i['instruction']).split()[0] for i in region)
        if sum(v for op, v in counts.items() if op.startswith('IMMA.')) != 64:
            continue  # only the two complete G128 mainloops, not tiny tail loops
        kind = 'integer' if not counts.get('I2F', 0) else 'fp32_fallback'
        max_live = max(i['live_gpr'] for i in region)
        loops.append(dict(kind=kind, begin_pc=hex(region[0]['pc']), end_pc=hex(region[-1]['pc']),
            static_instructions=len(region), max_live_gpr=max_live, opcode_counts=dict(sorted(counts.items())),
            peak_pcs=[hex(i['pc']) for i in region if i['live_gpr'] == max_live]))
    if len(loops) != 2 or {r['kind'] for r in loops} != {'integer', 'fp32_fallback'}:
        raise ValueError('expected one integer and one FP32 mainloop')
    return dict(symbol=symbol, allocated_gpr=int(allocated[1]),
        function_max_live_gpr=max(i['live_gpr'] for i in instructions), loops=loops,
        cta_threads=128, register_only_four_cta_limit_per_thread=65536 // (4 * 128),
        interpretation='static_binary_live_range_not_runtime_timing_or_universal_register_lower_bound')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('reports/o378_roof_v67_codegen'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT):
        p.error('fresh repository output required')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((a.baseline / 'codegen.json').read_text())
    cubin = a.baseline / 'o78_fullk.cubin'
    if digest(cubin) != receipt['cubin_sha256']:
        raise ValueError('baseline binary SHA mismatch')
    for name, sha in receipt['sources'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('baseline source drift: ' + name)
    tool = '/usr/local/cuda-12.8/bin/nvdisasm'
    command = [tool, '--print-code', '--life-range-mode', 'count', str(cubin)]
    text = subprocess.check_output(command, text=True)
    result = analyze(text)
    result.update(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=digest(Path(__file__)), baseline_cubin_sha256=receipt['cubin_sha256'],
        command=command, nvdisasm_version=subprocess.check_output([tool, '--version'], text=True),
        default_changed=False, new_kernel_built=False, new_runtime_measurements=False)
    a.output.mkdir(parents=True)
    artifact = a.output / 'o78_fullk_liveness.txt'
    artifact.write_text(text)
    result['liveness_sha256'] = digest(artifact)
    (a.output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

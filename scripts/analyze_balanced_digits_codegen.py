#!/usr/bin/env python3
"""Replay v100 compile proof; never infer measured speed or MSE from SASS."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from compare_a100_codegen import compare
from inspect_o78_register_liveness import analyze
from probe_o78_balanced_digits_codegen import (
    CONTROL, ROOT, SYMBOL, generated_headers, runtime_gate, scalar_proof)
from probe_roof_fullk_integer_codegen import static_entries


def loop_sequence(sass, symbol, loop):
    block = next(b for b in re.split(r'(?=Function\s*:\s*)',sass)
                 if b.startswith('Function : '+symbol+'\n'))
    begin, end = int(loop['begin_pc'],16), int(loop['end_pc'],16)
    instructions = []
    for line in block.splitlines():
        pc = re.match(r'\s*/\*([0-9a-fA-F]+)\*/',line)
        if pc and begin <= int(pc[1],16) <= end:
            text = re.sub(r'/\*.*?\*/','',line).strip()
            instructions.append(text)
    if len(instructions) != loop['static_instructions']:
        raise ValueError('loop SASS/liveness instruction coverage mismatch')
    return instructions


def summarize(directory):
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = json.loads((directory/'codegen.json').read_text())
    for name, digest in receipt['sources'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('compiler source drift: '+name)
    for name, digest in receipt['artifact_sha256'].items():
        if name.endswith('.cubin'):
            if digest != receipt['cubin_sha256']:
                raise ValueError('binary receipt drift')
            # Binary is retained in the raw archive, not published in Git.
            if (directory/name).is_file() and sha(directory/name) != digest:
                raise ValueError('local binary fingerprint mismatch')
        elif sha(directory/name) != digest:
            raise ValueError('curated artifact drift: '+name)
    for name, text in generated_headers().items():
        if (directory/name.replace('_cuh','.cuh')).read_text() != text:
            raise ValueError('generated candidate body drift: '+name)
    if receipt['scalar_proof'] != scalar_proof():
        raise ValueError('exhaustive source-code proof drift')
    sass = (directory/'o78_balanced_digits.sass').read_text()
    symbols = {CONTROL,SYMBOL}
    if static_entries(sass,'^(?:'+'|'.join(sorted(symbols))+')$',symbols) != receipt['entries']:
        raise ValueError('native audit receipt drift')
    lives = {symbol:analyze((directory/'liveness.txt').read_text(),symbol) for symbol in symbols}
    if lives != receipt['liveness']:
        raise ValueError('liveness receipt drift')
    baseline = ROOT/'docs/evidence/a100_o378_roof_v78/reports/o378_roof_v78_codegen/o78_eight_chain.sass'
    comparison = compare(baseline.read_text(),sass,'^'+CONTROL+'$')
    if comparison != receipt['control_comparison'] or not comparison['passed']:
        raise ValueError('exact old control encoding check failed')
    gate = runtime_gate(lives[CONTROL],lives[SYMBOL])
    if gate != receipt['runtime_gate']:
        raise ValueError('predeclared gate drift')
    rows, sequence_checks = {}, {}
    for kind in ('integer','fp32_fallback'):
        old = next(x for x in lives[CONTROL]['loops'] if x['kind'] == kind)
        new = next(x for x in lives[SYMBOL]['loops'] if x['kind'] == kind)
        sequences = [loop_sequence(sass,symbol,loop) for symbol,loop in ((CONTROL,old),(SYMBOL,new))]
        normalized = [s.replace('IMMA.16864.U4.S4','IMMA.16864.S4.S4') for s in sequences[0]]
        sequence_checks[kind] = dict(equal_except_low_MMA_signedness=normalized==sequences[1],
            scope='same_entry_mainloop_text_including_operands_not_raw_machine_bit_equality',
            old_normalized_sequence_sha256=hashlib.sha256('\n'.join(normalized).encode()).hexdigest(),
            candidate_sequence_sha256=hashlib.sha256('\n'.join(sequences[1]).encode()).hexdigest())
        local = lambda loop: sum(v for op,v in loop['opcode_counts'].items() if op.split('.')[0] in ('LDL','STL'))
        rows[kind] = dict(old_instructions=old['static_instructions'],candidate_instructions=new['static_instructions'],
            old_peak_live_gpr=old['max_live_gpr'],candidate_peak_live_gpr=new['max_live_gpr'],
            old_hot_local_instructions=local(old),candidate_hot_local_instructions=local(new),
            old_native_u4s4=old['opcode_counts'].get('IMMA.16864.U4.S4',0),
            old_native_s4s4=old['opcode_counts'].get('IMMA.16864.S4.S4',0),
            candidate_native_s4s4=new['opcode_counts'].get('IMMA.16864.S4.S4',0),
            old_ldsm=old['opcode_counts']['LDSM.16.M88.4'],candidate_ldsm=new['opcode_counts']['LDSM.16.M88.4'])
    return dict(scope='v100_hash_bound_compile_replay_no_GPU_launch_or_performance_claim',
        source_commit=receipt['source_commit'],compiler_receipt_sha256=sha(directory/'codegen.json'),
        analyzer_sha256=sha(Path(__file__)),scalar_proof=receipt['scalar_proof'],
        control_machine_encoding_passed=comparison['passed'],
        old_allocated_gpr=lives[CONTROL]['allocated_gpr'],candidate_allocated_gpr=lives[SYMBOL]['allocated_gpr'],
        cta_tile=receipt['cta_tile'],threads=receipt['threads'],shared_bytes=receipt['shared_bytes'],
        rows=rows,sequence_checks=sequence_checks,runtime_gate=gate,
        performance_measured=False,MSE_measured=False,production_default_changed=False,o3_changed=False,
        conclusion='stop_after_compile_no_reduced_instruction_or_residency_budget'
                   if not gate['worth_runtime_validation'] else 'requires_separate_GPU_acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    result = summarize(args.input.resolve())
    if args.output:
        out = args.output.resolve()
        if not out.is_relative_to(ROOT) or out.exists():
            parser.error('fresh repository output required')
        out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
